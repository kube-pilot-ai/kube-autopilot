"""Approval-only access to the MCP remediation tools."""
import hashlib
import hmac
import json
import os
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field


class DeploymentChange(BaseModel):
    operation: Literal["create", "patch", "delete"]
    namespace: str
    deployment: str
    container: str | None = Field(description="Container name for creation or patch, or null for deletion.")
    memory_request: str | None = Field(description="New memory request in Mi or Gi, or null.")
    memory_limit: str | None = Field(description="New memory limit in Mi or Gi, or null.")
    image: str | None = Field(description="Exact image for creation or replacement, or null. Image patches are separate from memory patches.")


class TroubleshootingResult(BaseModel):
    answer: str
    change: DeploymentChange | None = Field(description="A concrete Deployment creation, deletion, or evidence-backed patch for review, or null.")


def _result(tool, arguments):
    payload = tool.invoke(arguments)
    if isinstance(payload, list):
        payload = next(block["text"] for block in payload if block.get("type") == "text")
    return json.loads(payload)


class Remediation:
    def __init__(self, preview, apply, secret):
        self.preview = preview
        self.apply = apply
        self.secret = secret

    def prepare(self, state):
        plan = _result(self.preview, {
            key: value for key, value in state["candidate_change"].items() if value is not None
        })
        if "error" in plan:
            return {"messages": [self._answer(state, "No change submitted for approval. " + plan["error"])]}
        return {"pending_remediation": {"id": str(uuid4()), "change": plan}}

    @staticmethod
    def review(state):
        pending = state["pending_remediation"]
        decision = interrupt({"kind": "deployment_remediation", "plan_id": pending["id"]})
        if decision["plan_id"] != pending["id"] or decision["approved"] is not True:
            return {"pending_remediation": None,
                    "messages": [Remediation._answer(state, "Remediation declined. The cluster was left as it was.")]}
        return {"approved_plan_id": pending["id"]}

    def execute(self, state):
        pending = state["pending_remediation"]
        if not pending or state.get("approved_plan_id") != pending["id"]:
            raise RuntimeError("An approved plan is required before execution.")
        approval = json.dumps(pending["change"], sort_keys=True, separators=(",", ":"))
        signature = hmac.new(self.secret.encode(), approval.encode(), hashlib.sha256).hexdigest()
        result = _result(self.apply, {"approval": approval, "signature": signature})
        change = pending["change"]
        target = f'{change["namespace"]}/{change["deployment"]}'
        if "error" in result:
            message = f'Approved change was rejected for {target}. {result["error"]}'
        elif result["status"] == "deletion_requested":
            message = f'Kubernetes accepted deletion of Deployment {target}. Dependent ReplicaSets and pods are cleaned up in the background.'
        elif result["status"] == "already_absent":
            message = f'Deployment {target} is already absent.'
        elif result["status"] == "created":
            message = f'Created the approved Deployment {target}.'
        elif result["status"] == "already_created":
            message = f'Deployment {target} already exists from this approval.'
        elif result["status"] == "already_present":
            message = f'The approved settings are already present on {target}. No patch was needed.'
        else:
            message = (f'Applied the approved change to Deployment {target}, container {change["container"]}. '
                       f'Kubernetes accepted generation {result["generation"]}.')
        if rollout := result.get("rollout"):
            observed = rollout["observed_generation"]
            desired = rollout["desired"]
            if observed is None or observed < result["generation"]:
                message += " The Deployment controller has yet to observe this generation."
            else:
                message += f' Rollout status: {rollout["updated"]}/{desired} updated, {rollout["available"]}/{desired} available.'
            message += f' Check recovery with: kubectl rollout status deployment/{change["deployment"]} -n {change["namespace"]}'
        return {"pending_remediation": None, "approved_plan_id": None,
                "remediation_outcome": message, "messages": [self._answer(state, message)]}

    @staticmethod
    def _answer(state, message):
        return AIMessage(content=state["messages"][-1].text + "\n\n" + message, name="troubleshooting_agent")


def split_tools(tools):
    preview_name = "preview_deployment_remediation"
    apply_name = "apply_deployment_remediation"
    by_name = {tool.name: tool for tool in tools}
    read_only_names = {
        "list_nodes", "list_namespaces", "list_pods", "list_deployments", "list_services",
        "get_pod_logs_and_events", "get_pod_details", "get_deployment_details", "get_node_details",
    }
    read_tools = [tool for tool in tools if tool.name in read_only_names]
    secret = os.environ.get("REMEDIATION_SIGNING_KEY")
    remediation = None
    if secret and preview_name in by_name and apply_name in by_name:
        remediation = Remediation(by_name[preview_name], by_name[apply_name], secret)
    return read_tools, remediation

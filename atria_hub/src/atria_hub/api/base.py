from atria_hub.client import AtriaHubClient


class BaseApi:
    def __init__(self, client: AtriaHubClient):
        self._client = client

    def get_commit_sha(self, repo_id: str, branch: str) -> str:
        import lakefs

        return (
            lakefs.repository(repo_id, client=self._client.lakefs_client)
            .branch(branch)
            .get_commit()
            .id[:7]
        )

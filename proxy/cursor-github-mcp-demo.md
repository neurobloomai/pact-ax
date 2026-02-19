# Cursor ↔ GitHub MCP — End-to-End Demo

> **Important:** The old `@modelcontextprotocol/server-github` npm package is **deprecated as of April 2025**.
> Use the configs below instead.

---

## 1. Pick Your Transport

### Option A — Remote HTTP (Recommended, zero dependencies)

**Requires:** Cursor v0.48.0+, GitHub PAT

```json
// ~/.cursor/mcp.json  (global)  OR  .cursor/mcp.json  (project-root)
{
  "mcpServers": {
    "github": {
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_PAT"
      }
    }
  }
}
```

### Option B — Local Docker (no cloud calls)

**Requires:** Docker Desktop running

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_PAT"
      }
    }
  }
}
```

### Option C — Keep token out of the file (best practice)

Set the token in your shell profile first:

```bash
# ~/.zshrc or ~/.bashrc
export GITHUB_PAT=ghp_xxxxxxxxxxxxxxxxxxxx
```

Then in `mcp.json` reference it:

```json
{
  "mcpServers": {
    "github": {
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer ${env:GITHUB_PAT}"
      }
    }
  }
}
```

---

## 2. PAT Scopes Needed

| What you want to do | Required scopes |
|---|---|
| Read public repos, issues, PRs | `public_repo` |
| Read private repos | `repo` |
| Create/edit issues & PRs | `repo` |
| Read GitHub Actions runs | `workflow` |
| Read org-level data | `read:org` |

Go to **GitHub → Settings → Developer settings → Personal access tokens** (classic or fine-grained both work).

---

## 3. Activate in Cursor

1. **Edit the file:** `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in your project root.
2. **Restart Cursor** (Cmd+Shift+P → "Reload Window" also works).
3. **Verify:** Open the Cursor chat sidebar → click the **Tools** icon (⚙) → you should see `github` listed with a green dot.

If you see a red dot or no dot: check the token, check Cursor version (`Cursor → About Cursor`).

---

## 4. What Tools Become Available

Once connected, Cursor's AI gains these tools (the server exposes ~40+ total):

| Category | Example tools |
|---|---|
| **Repos** | `search_repositories`, `get_file_contents`, `list_commits` |
| **Issues** | `list_issues`, `create_issue`, `add_issue_comment` |
| **Pull Requests** | `list_pull_requests`, `create_pull_request`, `merge_pull_request`, `get_pull_request_diff` |
| **Branches** | `list_branches`, `create_branch`, `delete_branch` |
| **Files** | `create_or_update_file`, `push_files` |
| **Search** | `search_code`, `search_issues` |
| **Actions** | `list_workflow_runs`, `get_workflow_run` |

---

## 5. Demo Prompts — Copy-Paste into Cursor Chat

Open Cursor chat (Cmd+L), make sure **Agent** mode is selected, then try:

### 5.1 Read a repo

```
List the last 5 commits on the main branch of <owner>/<repo>.
Summarize what changed.
```

### 5.2 Explore open issues

```
Show me all open issues in <owner>/<repo> labeled "bug".
Group them by severity if mentioned.
```

### 5.3 Create a branch + file

```
In <owner>/<repo>, create a branch called "demo/mcp-test",
then create a file at "mcp-test/hello.md" with the content
"Hello from Cursor MCP demo — {today's date}".
```

### 5.4 Open a pull request

```
Open a pull request in <owner>/<repo> from branch "demo/mcp-test"
to "main" with title "MCP demo: add hello file" and a short description.
```

### 5.5 Code search

```
Search for all usages of "useEffect" across <owner>/<repo> and
summarize the top 3 files where it's used most.
```

### 5.6 Review a PR diff

```
Get the full diff of PR #<number> in <owner>/<repo>
and flag any obvious issues.
```

---

## 6. Verifying the Round-Trip (smoke test)

Run this prompt — it exercises read + write + read in one shot:

```
Do the following in sequence and confirm each step:
1. List the 3 most recent open issues in <owner>/<repo>
2. Create a new issue titled "MCP smoke test — can delete" with body "Testing Cursor ↔ GitHub MCP"
3. Immediately close the issue you just created
```

If all three steps complete with real GitHub issue numbers, **your MCP flow is working end-to-end.**

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Red dot on `github` server | Token invalid or expired — regenerate PAT |
| "Tool not found" error | Restart Cursor after editing `mcp.json` |
| Works globally but not in project | Project `.cursor/mcp.json` is overriding global; merge configs |
| Docker option hangs | Docker Desktop not running, or image not pulled (`docker pull ghcr.io/github/github-mcp-server`) |
| Remote URL 401 | PAT missing required scopes (see §2) |
| Cursor version mismatch | Remote HTTP needs v0.48+; check `Cursor → About` |

---

## 8. Project-Level vs Global Config

```
~/.cursor/mcp.json          ← applies to ALL projects (GitHub, web search, etc.)
<project-root>/.cursor/mcp.json ← applies only to THIS project (overrides global for that server key)
```

**Tip:** Commit `.cursor/mcp.json` to the repo so teammates get the same tools. Add `GITHUB_PAT` to `.env` (gitignored) and reference it with `${env:GITHUB_PAT}`.

---

*Sources: [github/github-mcp-server](https://github.com/github/github-mcp-server) · [Cursor install guide](https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-cursor.md)*

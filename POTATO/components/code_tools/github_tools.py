import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()

AUTH_TOKEN = str(os.getenv("GITHUB_TOKEN")).strip()
print("Loaded GITHUB_TOKEN from .env.local: ", AUTH_TOKEN)

# https://docs.github.com/en/rest/using-the-rest-api/getting-started-with-the-rest-api?apiVersion=2022-11-28#http-method
# https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repositories-for-the-authenticated-user
# https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repositories-for-a-user
# https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28
# https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28#download-a-repository-archive-zip
# https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28#create-or-update-file-contents
# https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28
# https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#get-a-repository
# 

endpoints = {
    "create_repo":"",
    "get_repo":"https://api.github.com/user/repos/{repo_name}",
    "list_repos":"https://api.github.com/user/repos",
    "update_repo":"",
    "delete_repo":"",
    "download_zip":"https://api.github.com/repos/{owner}/{repo}/zipball/{ref}", # ref can be branch, tag or commit sha
    "upload_file":"https://api.github.com/repos/{owner}/{repo}/contents/{path}", # path is the file path in the repo
    "delete_file":"",
    "search_repo":"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc", # query can be any search term
    "update_file_content":"https://api.github.com/repos/{owner}/{repo}/contents/{path}", # path is the file path in the repo, content is the new content of the file
    "get_repo_files":"https://api.github.com/repos/{owner}/{repo}/contents",
    "create_issue":"https://api.github.com/repos/{owner}/{repo}/issues", # title and body are required
    "list_issues":"https://api.github.com/repos/{owner}/{repo}/issues", # state can be open, closed or all
    "get_repo_info":"https://api.github.com/repos/{owner}/{repo}", # owner and repo are required
    # and much more

}

headers = {
    "Authorization": f"token {api_key}",
    "Accept": "application/vnd.github.v3+json"
}

# make api request function, uses dict keys
def github_api_request(api_key: str, method: str, url: str, data=None, headers=None) -> dict:
    # to implement
    return 0

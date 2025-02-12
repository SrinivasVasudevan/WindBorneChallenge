import requests
url = "https://windbornesystems.com/career_applications.json"
req = {
  "career_application": {
    "name": "Srinivas Vasudevan",
    "email": "svasude7@ncsu.edu",
    "role": "Atlas Software Intern",
    "resume_url": "https://srinivasvasudevan.github.io/Resume/Resume_Srinivas_Vasudevan.pdf",
        "submission_url": "I just wanted to clarify the terminology \" Host your creation \" for this challenge. Would you like the challenge submission hosted on a web server, or would a link to the GitHub repo with instructions to run the project suffice? If you would like the former, my home server isn't capable of Wake-on-Lan, meaning the website has a non-zero chance of going down. In this case, information about the possible time at which evaluations might happen would be of great help. You can contact me via my email address: svasude7@ncsu.edu. \n Having fun progressing towards a final submission. \n Looking forward to your reply. Thanks!. Curious about my wip: ",
    "query_body": "I just wanted to clarify the terminology \" Host your creation \" for this challenge. Would you like the challenge submission hosted on a web server, or would a link to the GitHub repo with instructions to run the project suffice? If you would like the former, my home server isn't capable of Wake-on-Lan, meaning the website has a non-zero chance of going down. In this case, information about the possible time at which evaluations might happen would be of great help. You can contact me via my email address: svasude7@ncsu.edu. \n Having fun progressing towards a final submission. \n Looking forward to your reply. Thanks!",

  }
}
resp = requests.post(url, json = req)
print(resp, resp.text)

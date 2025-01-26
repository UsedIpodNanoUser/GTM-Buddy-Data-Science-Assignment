# GTM-Buddy-Data-Science-Assignment

to run the docker image, following commands must be entered into cmd after navigating to directory

docker build -t dockerfile .
docker run -p 8000:8000 dockerfile

use swagger UI to interact with the api with command:

uvicorn main:app --reload

Then open browser: 

http://localhost:8000/docs

Click "Try it out"
Enter JSON snippet
Execute request

output should be as expected

or use curl command

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text_snippet": "We are comparing prices with BiggyCorp"}'

import yaml
import requests
import json
from jsonschema import validate, ValidationError

# Load the YAML file
with open('.github/workflows/project-tds-virtual-ta-promptfoo.yaml', 'r') as file:
    test_data = yaml.safe_load(file)

# Function to send HTTP request and get the response
def send_request(url, method, headers, body):
    if method == 'POST':
        response = requests.post(url, headers=headers, data=body)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    return response

# Function to validate JSON schema
def validate_json_schema(response_json, schema):
    try:
        validate(instance=response_json, schema=schema)
        return True, ""
    except ValidationError as e:
        return False, str(e)

# Function to check if a substring is in a string
def contains(substring, string):
    return substring in string

# Function to run the tests
def run_tests(test_data):
    results = []
    for test in test_data['tests']:
        vars = test['vars']
        url = test_data['providers'][0]['config']['url']
        method = test_data['providers'][0]['config']['method']
        headers = test_data['providers'][0]['config']['headers']
        body_template = test_data['providers'][0]['config']['body']

        # Substitute variables in the body template
        body = body_template
        for key, value in vars.items():
            body = body.replace(f"{{{{ {key} }}}}", value)

        # Send the request
        response = send_request(url, method, headers, body)
        response_json = response.json()

        # Run assertions
        test_result = {"test": vars, "assertions": []}
        for assertion in test['assert']:
            if assertion['type'] == 'is-json':
                schema = assertion['value']
                valid, error = validate_json_schema(response_json, schema)
                test_result['assertions'].append({
                    "type": "is-json",
                    "result": valid,
                    "error": error
                })
            elif assertion['type'] == 'contains':
                substring = assertion['value']
                string = json.dumps(response_json)
                valid = contains(substring, string)
                test_result['assertions'].append({
                    "type": "contains",
                    "result": valid,
                    "substring": substring
                })
            elif assertion['type'] == 'llm-rubric':
                expected_value = assertion['value']
                actual_value = eval(assertion['transform'], {}, {"output": response_json})
                valid = expected_value in actual_value
                test_result['assertions'].append({
                    "type": "llm-rubric",
                    "result": valid,
                    "expected": expected_value,
                    "actual": actual_value
                })
            else:
                raise ValueError(f"Unsupported assertion type: {assertion['type']}")

        results.append(test_result)
    return results

# Run the tests and print the results
results = run_tests(test_data)
for result in results:
    print(f"\nTest: {result['test']}")
    for assertion in result['assertions']:
        print(f"  Assertion: {assertion['type']}")
        print(f"    Result: {'✅ PASS' if assertion['result'] else '❌ FAIL'}")
        if 'error' in assertion:
            print(f"    Error: {assertion['error']}")
        if 'substring' in assertion:
            print(f"    Substring: {assertion['substring']}")
        if 'expected' in assertion:
            print(f"    Expected: {assertion['expected']}")
        if 'actual' in assertion:
            print(f"    Actual: {assertion['actual']}")

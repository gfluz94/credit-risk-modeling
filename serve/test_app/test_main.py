from fastapi.testclient import TestClient
from app.main import app


class TestApp(object):

    client = TestClient(app)

    def test_homeStatusCode200(self):
        # OUTPUT
        response = self.client.get("/")

        # EXPECTED
        success_status_code = 200

        # ASSERT
        assert response.status_code == success_status_code

    def test_homeReturnsJSON(self):
        # OUTPUT
        response = self.client.get("/")

        # EXPECTED
        json_response = "Home"

        # ASSERT
        assert response.json() == json_response

    def test_predictNoPOSTRequest(self):
        # OUTPUT
        response = self.client.get("/predict")

        # EXPECTED
        error_code = 405

        # ASSERT
        assert response.status_code == error_code

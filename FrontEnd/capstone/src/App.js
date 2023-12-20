import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import LoginPage from "./components/LoginPage/LoginPage";
import Register from "./components/Register/Register";
import Dashboard from "./components/Dashboard/Dashboard";
import SummaryPage from "./components/SummaryPage/SummaryPage";
import ConsentPage from "./components/ConsentPage/ConsentPage";
import DeliveryPage from "./components/DeliveryPage/DeliveryPage";
function App() {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route exact path="/loginpage" element={<LoginPage />}></Route>
          <Route exact path="/register" element={<Register />}></Route>
          <Route exact path="/dashboard" element={<Dashboard />}></Route>
          <Route exact path="/summarypage" element={<SummaryPage />}></Route>
          <Route exact path="/consentpage" element={<ConsentPage />}></Route>
          <Route exact path="/deliverypage" element={<DeliveryPage />}></Route>
        </Routes>
      </Router>
    </div>
  );
}

export default App;

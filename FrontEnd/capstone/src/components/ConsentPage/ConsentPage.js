import "./ConsentPage.css";
import axios from "axios";
import React, { useEffect, useState } from "react";
import logo from "./logo5.png";
import { useResolvedPath } from "react-router-dom";

const ConsentPage = () => {
  var usr_nm = localStorage.getItem("username");

  const [user, setUser] = useState({
    digsign: "",
  });
  /*
  useEffect(() => {
    init_func();
  }, []);
*/
  const logout_func = () => {
    window.location.href = "loginpage";
  };

  const handleChange = (event) => {
    var name1 = event.target.id;
    var value1 = event.target.value;

    setUser({
      ...user,
      [name1]: value1,
    });
    console.log(user.digsign);
  };

  const handleSubmit = () => {
    console.log(user.digsign);
    console.log(usr_nm);
    if (user.digsign != usr_nm) {
      alert("Your Digital Signature is not correct !");
      return;
    }

    console.log(user);
    let headers = new Headers();

    headers.append("Content-Type", "application/json");
    headers.append("Accept", "application/json");
    headers.append("Origin", "http://localhost:1086");
    headers.append("Access-Control-Allow-Origin", "*");

    const url = `http://localhost:9995/updateConsent?type=register&username=${usr_nm}&lastone=last`;

    axios
      .post(url, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        if (response.status == 200) {
          alert("Your Consent is recorded !");
          window.location.href = "/dashboard";
        } else {
          alert("Server Error: could not update consent status !");
        }
      })
      .catch((error) => {
        console.error("File upload failed:", error);
      });
  };

  return (
    <div id="highest">
      <div id="rootdiv">
        <div id="topmenu">
          <div id="tagname">LearnCraft</div>
          <div className="dropdown">
            Hello, {usr_nm}!
            <div className="dropdown-content" onClick={logout_func}>
              <p>Log Out</p>
            </div>
          </div>
        </div>
        <div id="inlin">
          <h1 id="getterlineconsent">Consent Form</h1>
        </div>

        <div id="lowerconsent">
          <div id="consent">
            <h1 id="consentlineconsent">
              I confirm that i won't misuse this product and won't perform any
              kind of indentity-theft. I will use this product
              <br />{" "}
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;only
              to generate AV summaries for educational purpose. I am aware that
              these can lead
              <br />
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              to legal actions against me in case, i misuse the product.
            </h1>
          </div>
          <table id="sigtable">
            <tr>
              <td id="sigtabletd">
                <h1 id="secondline">
                  Enter your Username in the Adjacent field <br />
                  as your Digital Signature
                </h1>
              </td>
              <td id="sigtabletd">
                <input
                  type="text"
                  id="digsign"
                  className="inp_styles"
                  placeholder="Your Digital Signature"
                  onChange={handleChange}
                ></input>
              </td>
            </tr>
          </table>

          <div>
            <button
              id="generate"
              className="but_stylererdash3"
              onClick={handleSubmit}
            >
              Submit
            </button>
          </div>
        </div>

        <div id="footerpggg">Copyright &copy; 2023 - LearnCraft</div>
      </div>
    </div>
  );
};
export default ConsentPage;

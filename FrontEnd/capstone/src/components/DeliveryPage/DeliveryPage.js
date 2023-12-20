import "./DeliveryPage.css";
import axios from "axios";
import React, { useEffect, useState } from "react";
import { useResolvedPath } from "react-router-dom";

const DeliveryPage = () => {
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

          <div id="menus1user">
            <a id="a22" href="/dashboard">
              Dashboard
            </a>
          </div>
        </div>

        <div id="inlin">
          <h1 id="getterlinedelivery">AV Summary</h1>
        </div>

        <div id="lowerconsent">
          <div id="consent">
            <h1 id="consentlinedelivery">
              Please find the generated AV summary below !!
            </h1>
          </div>
        </div>

        <div id="videoshower">
          <video width="950" height="475" controls>
            <source
              src="http://localhost:9995/videos/FinalVideo.mp4"
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
        </div>
        <div id="footerpggg">Copyright &copy; 2023 - LearnCraft</div>
      </div>
    </div>
  );
};
export default DeliveryPage;

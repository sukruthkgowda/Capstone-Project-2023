import "./Dashboard.css";
import axios from "axios";
import React, { useEffect, useState } from "react";
import logo from "./logo5.png";
import { useResolvedPath } from "react-router-dom";

const Dashboard = () => {
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
    if (selectedVideo === null || selectedTextFile === null) {
      alert("Please Choose Both files !");
      return;
    }

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

    const url = `http://localhost:9995/getuserpreferencesong?type=register&username=${usr_nm}&lastone=last`;
    const formData = new FormData();
    formData.append("file", selectedVideo);

    axios
      .post("http://localhost:9995/uploadVideo", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        console.log("File uploaded:", response.data);

        const formDataNew = new FormData();
        formDataNew.append("file", selectedTextFile);
        axios
          .post("http://localhost:9995/uploadTextFile", formDataNew, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          })
          .then((response) => {
            alert("Files Uploaded !");
            console.log("File uploaded:", response.data);

            document.getElementById("loading").style.display = "block";

            axios
              .get("http://localhost:9995/generatesummary", {
                headers: {
                  "Content-Type": "multipart/form-data",
                },
              })
              .then((responsee) => {
                document.getElementById("loading").style.display = "none";
                console.log("Generated Summary:", responsee.data);
                alert("Summary Generated");
                window.location.href = "/summarypage";
              })
              .then((errorr) => {
                console.log("Summary Generation Error");
              });
          })
          .catch((error) => {
            console.error("File upload failed:", error);
          });
      })
      .catch((error) => {
        console.error("File upload failed:", error);
      });
  };

  const scroll_func = () => {
    window.scrollTo({ top: 775, behavior: "smooth" });
  };

  const [selectedVideo, setSelectedVideo] = useState(null);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    console.log("Selected File:", file);
    setSelectedVideo(file);
  };

  const [selectedTextFile, setSelectedTextFile] = useState(null);

  const handleTextFileChange = (event) => {
    const file = event.target.files[0];
    console.log("Selected File:", file);
    setSelectedTextFile(file);
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
          <h1 id="getterline">
            Excited to Ease Up
            <br /> Your Hectic Work ?
          </h1>
          <h2 id="secondline">
            Create the Lectures here to save your time <br />
            and be more Productive by Automating <br />
            the Repeatetive and Tiring work!
          </h2>
          <button
            onClick={scroll_func}
            id="getstartedbtn"
            className="but_stylerergetstarted"
          >
            Get Started
          </button>
        </div>

        <div id="dashimage">
          <img src={logo} alt="Logo" height="550px" width="950px" />;
        </div>
        <h1 id="middlehead">Create Your Lectures!</h1>
        <div id="middlefortable">
          <table id="uploadtable">
            <tr id="uploadtabletr">
              <td id="uploadtabletd">
                <h1 id="secondline">
                  Upload your Video <br />
                  ( Make sure you are <br />
                  interacting in the video )
                </h1>
              </td>
              <td id="uploadtabletd">
                <input
                  type="file"
                  accept="video/*"
                  id="video-input"
                  onChange={handleVideoChange}
                />
              </td>
            </tr>
            <tr id="uploadtabletr">
              <td id="uploadtabletd">
                <h1 id="secondline">
                  Upload a text file containing <br />
                  the text theory
                </h1>
              </td>
              <td id="uploadtabletd">
                <input
                  type="file"
                  accept="text/*"
                  id="textfile-input"
                  onChange={handleTextFileChange}
                />
              </td>
            </tr>
          </table>
        </div>

        <div id="lowerconsent">
          <div id="consent">
            <h1 id="consentline">
              I confirm that the video uploaded is mine and i am not performing
              identity-theft. I am aware <br />{" "}
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; that
              these can lead to legal actions against me in case, i misuse the
              product.
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
              Generate
            </button>
          </div>

          <div id="loading">
            <div class="spinner"></div>
            <h1 id="loadingline">Loading</h1>
          </div>
        </div>

        <div id="footerpggg">Copyright &copy; 2023 - LearnCraft</div>
      </div>
    </div>
  );
};
export default Dashboard;

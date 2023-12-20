import "./SummaryPage.css";
import axios from "axios";
import React, { useEffect, useState } from "react";
import logo from "./logo5.png";
import { useResolvedPath } from "react-router-dom";

const SummaryPage = () => {
  var usr_nm = localStorage.getItem("username");

  const [user, setUser] = useState({
    summaryinput: "",
  });
  useEffect(() => {
    init_func();
  }, []);

  const init_func = () => {
    let headers = new Headers();

    headers.append("Content-Type", "application/json");
    headers.append("Accept", "application/json");

    headers.append("Origin", "http://localhost:1086");
    headers.append("Access-Control-Allow-Origin", "*");

    const url = `http://localhost:9995/getgeneratedsummary`;

    axios.post(url).then((response) => {
      console.log(response.data);
      user.summaryinput = response.data;
      console.log(user.summaryinput);
      const tarea = document.getElementById("summaryinput");
      tarea.value = response.data;
    });
  };

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

  const handleSubmit = (e) => {
    e.preventDefault();

    if (user.summaryinput === "") {
      alert("The summary cannot be empty!");
    } else {
      let headers = new Headers();

      headers.append("Content-Type", "application/json");
      headers.append("Accept", "application/json");

      headers.append("Origin", "http://localhost:1086");
      headers.append("Access-Control-Allow-Origin", "*");

      const url = `http://localhost:9995/updatesummary?type=register&firstone=fist&summary=${user.summaryinput}&lastone=last`;
      axios.post(url).then((response) => {
        console.log(response.status);
        if (response.status == 200) {
          alert("You have updated summary successfully !!");
          document.getElementById("summarypageloading").style.display = "block";
          //window.location.href = "loginpage";
          const urlTTS = `http://localhost:9995/generateTTSoutput?type=register&firstone=fist&lastone=last`;
          axios.get(urlTTS).then((responseee) => {
            console.log(responseee.status);
            if (responseee.status == 200) {
              //alert("TTS output generated successfully !!");

              const urldeepfake = `http://localhost:9995/generateDeepfakeOutput?type=register&firstone=fist&lastone=last`;
              axios.get(urldeepfake).then((responseeee) => {
                console.log(responseeee.status);
                document.getElementById("summarypageloading").style.display =
                  "none";
                if (responseeee.status == 200) {
                  alert("Deepfake output generated successfully !!");
                  window.location.href = "deliverypage";
                } else {
                  alert("Error in generating Deepfake output !!");
                }
              });

              //window.location.href = "loginpage";
            } else {
              alert("Error in generating TTS output !!");
            }
          });
        } else {
          alert("Error in updating Summary!!");
        }
      });
    }
  };

  const handleSubmitForAPI = (e) => {
    e.preventDefault();

    if (user.summaryinput === "") {
      alert("The summary cannot be empty!");
    } else {
      let headers = new Headers();

      headers.append("Content-Type", "application/json");
      headers.append("Accept", "application/json");

      headers.append("Origin", "http://localhost:1086");
      headers.append("Access-Control-Allow-Origin", "*");

      const url = `http://localhost:9995/updatesummary?type=register&firstone=fist&summary=${user.summaryinput}&lastone=last`;
      axios.post(url).then((response) => {
        console.log(response.status);
        if (response.status == 200) {
          alert("You have updated summary successfully !!");
          document.getElementById("summarypageloading").style.display = "block";
          //window.location.href = "loginpage";
          const urlTTS = `http://localhost:9995/generateTTSoutputFromAPI?type=register&username=${usr_nm}&firstone=fist&lastone=last`;
          axios.get(urlTTS).then((responseee) => {
            console.log(responseee.status);
            if (responseee.status == 200) {
              alert("TTS output from API generated successfully !!");

              const urldeepfake = `http://localhost:9995/generateDeepfakeOutput?type=register&firstone=fist&lastone=last`;
              axios.get(urldeepfake).then((responseeee) => {
                console.log(responseeee.status);
                document.getElementById("summarypageloading").style.display =
                  "none";
                if (responseeee.status == 200) {
                  alert("Deepfake output generated successfully !!");
                  window.location.href = "deliverypage";
                } else {
                  alert("Error in generating Deepfake output !!");
                }
              });

              //window.location.href = "loginpage";
            } else {
              alert("Error in generating TTS output !!");
            }
          });
        } else {
          alert("Error in updating Summary!!");
        }
      });
    }
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

        <div>
          <h1 id="middleheadsummary">Summary Check</h1>
          <h1 id="secondlinesummary">
            Please review the summary generated, and make any necessary changes
            required !
          </h1>
          <textarea
            id="summaryinput"
            placeholder="Your Summary"
            onChange={handleChange}
          ></textarea>
          <div id="generateflex">
            <button
              id="generatesummary"
              className="but_stylerersummary"
              onClick={handleSubmit}
            >
              Generate
            </button>
            <button
              id="generatesummary"
              className="but_stylerersummary2"
              onClick={handleSubmitForAPI}
            >
              Generate using API
            </button>
          </div>
          <div id="summarypageloading">
            <div class="summarypagespinner"></div>
            <h1 id="summarypageloadingline">Loading</h1>
          </div>
        </div>

        <div id="footerpggg">Copyright &copy; 2023 - LearnCraft</div>
      </div>
    </div>
  );
};
export default SummaryPage;

import React, { useState } from "react";
import axios from "axios";
import "./LoginPage.css";

const LoginForm = () => {
  const [user, setUser] = useState({
    user_name: "",
    passwordlogin: "",
  });

  const handleChange = (event) => {
    var name1 = event.target.id;
    var value1 = event.target.value;

    setUser({
      ...user,
      [name1]: value1,
    });
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log(user.user_name);
    console.log(user.password);
    if (user.user_name != "" && user.passwordlogin != "") {
      let headers = new Headers();

      headers.append("Content-Type", "application/json");
      headers.append("Accept", "application/json");

      headers.append("Origin", "http://localhost:1086");
      headers.append("Access-Control-Allow-Origin", "*");

      const url = `http://localhost:9995/login/?type=login&users=${user.user_name}&pass=${user.passwordlogin}`;

      axios
        .post(url, user)
        .then((response) => {
          console.log(response.status);
          if (response.status == 200) {
            alert("You have Logged in successfully!");
            localStorage.setItem("username", user.user_name);
            localStorage.setItem("password", user.password);

            window.location.href = "/dashboard";
          } else if (response.status == 201) {
            alert("You haven't registered yet, please register..");
          } else if (response.status == 234) {
            alert("You have Logged in successfully!");
            localStorage.setItem("username", user.user_name);
            localStorage.setItem("password", user.password);
            window.location.href = "/consentPage";
          } else {
            alert("Entered password is incorrect..");
          }
        })
        .then((html) => console.log(html));
    } else {
      alert("Empty Feilds Are Not Allowed..");
    }
  };

  const handleSubmit2 = (event) => {
    event.preventDefault();
    window.location.href = "/register";
  };

  return (
    <div class="parent clearfix">
      <div class="bg-illustration">
        <div id="brandname">
          <h2>LearnCraft</h2>
        </div>
        <div id="tagline">
          <h4>Crafting Learning With Innovation!</h4>
        </div>
        <div class="burger-btn">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>

      <div class="login">
        <div class="container23">
          <br />
          <h1>
            Log in to access <br />
            your account
          </h1>

          <div class="login-form">
            <form>
              <input
                type="text"
                placeholder="Username"
                onChange={handleChange}
                id="user_name"
              />
              <input
                type="password"
                placeholder="Password"
                onChange={handleChange}
                id="passwordlogin"
              />

              <button type="submit" id="submitbtn" onClick={handleSubmit}>
                LOG-IN
              </button>
              <br />
              <br />
              <hr width="85%" id="hr1" />
              <br />
              <br />
              <div id="signup">
                New Here?<span id="signupdown">Sign Up Now!</span>
              </div>
              <br />
              <button type="submit" id="submitbtn2" onClick={handleSubmit2}>
                Sign Up
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;

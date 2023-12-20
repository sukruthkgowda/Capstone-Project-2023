var express = require("express");
var app = express();
var url = require("url");
var axios = require("axios");
var MongoClient = require("mongodb").MongoClient;
const bodyParser = require("body-parser");
const formidable = require("formidable");
const http = require("http");
const FormData = require("form-data");
const { PassThrough } = require("stream");
const fs = require("fs");

const multer = require("multer");
const path = require("path");
var cors = require("cors");
app.use(cors());
var mysql = require("mysql");

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Specify the directory where you want to open the command prompt
const workingDirectory = "C:\\Users\\Asus\\Desktop\\Capstone\\TextSummarizer";

//const workingDirectory = 'C:\\path\\to\\your\\directory';
const virtualEnvPath =
  "C:\\Users\\Asus\\Desktop\\Capstone\\TextSummarizer\\myenv\\Scripts\\activate"; // Path to the activate script

const { exec } = require("child_process");

// Serve your video file located in a directory named "videos"
app.use("/videos", express.static(path.join(__dirname, "videos")));

app.get("/generatesummary", (req, res) => {
  const pythonCommand = "python runTextSummarizer.py"; // Replace with your Python command

  const pythonProcess = exec(pythonCommand, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error}`);
      return;
    }
    console.log(`Python script output: ${stdout}`);
    console.error(`Python script errors: ${stderr}`);
  });

  pythonProcess.on("exit", (code) => {
    console.log(`Python script exited with code ${code}`);
    // Here, you can continue with any JavaScript code that depends on the Python script being completed.
    res.status(200).send("Output Summary Generated");
  });
});

app.get("/generateDeepfakeOutput", (req, res) => {
  // Function to delete a file if it exists
  const inputtempPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\__temp__.mp4";
  const outputPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\output_video_with_audio.mp4";

  const deleteFileIfExists = (filePath) => {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${filePath}`);
    }
  };

  // Delete input and output files if they exist
  deleteFileIfExists(inputtempPath);
  deleteFileIfExists(outputPath);

  const pythonCommand = "python runDeepfake.py"; // Replace with your Python command

  const pythonProcess = exec(pythonCommand, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error}`);
      return;
    }
    console.log(`Python script output: ${stdout}`);
    console.error(`Python script errors: ${stderr}`);
  });

  pythonProcess.on("exit", (code) => {
    console.log(`Python script exited with code ${code}`);
    // Here, you can continue with any JavaScript code that depends on the Python script being completed.
    res.status(200).send("Deepfake Output Generated");
  });
});

// app.get("/generateTTSoutput", (req, res) => {
//   const pythonCommand = "python runTTS.py"; // Replace with your Python command

//   const pythonProcess = exec(pythonCommand, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`Error: ${error}`);
//       return;
//     }
//     console.log(`Python script output: ${stdout}`);
//     console.error(`Python script errors: ${stderr}`);
//   });

//   pythonProcess.on("exit", (code) => {
//     console.log(`Python script exited with code ${code}`);
//     // Here, you can continue with any JavaScript code that depends on the Python script being completed.
//     res.status(200).send("TTS Output Generated");
//   });
// });

app.get("/generateTTSoutputFromAPI", (req, res) => {
  // Function to delete a file if it exists
  const inputWavPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\uploads\\inputVideo.wav";
  const outputWavPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\out\\output.wav";

  const deleteFileIfExists = (filePath) => {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${filePath}`);
    }
  };

  // Delete input and output files if they exist
  deleteFileIfExists(inputWavPath);
  deleteFileIfExists(outputWavPath);

  var obj = url.parse(req.url, true).query;
  console.log(obj);
  console.log("username is ");
  console.log(obj.username);

  var sql = `SELECT * from users where username="${obj.username}"`;
  con.query(sql, function (err, res3, fields3) {
    if (err) {
      throw err;
    }

    console.log(res3);
    var gend = res3[0].gender;

    var pythonCommand = `python runTTSapi.py`; // Replace with your Python command

    if (gend === "FEMALE") {
      pythonCommand = `python runTTSapiforFemale.py`;
    }

    const pythonProcess = exec(pythonCommand, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${error}`);
        return;
      }
      console.log(`Python script output: ${stdout}`);
      console.error(`Python script errors: ${stderr}`);
    });

    pythonProcess.on("exit", (code) => {
      console.log(`Python script exited with code ${code}`);
      // Here, you can continue with any JavaScript code that depends on the Python script being completed.
      res.status(200).send("TTS Output using API Generated");
    });
  });
});

app.get("/generateTTSoutput", (req, res) => {
  // Function to delete a file if it exists
  const inputWavPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\uploads\\inputVideo.wav";
  const outputWavPath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\out\\output.wav";

  const deleteFileIfExists = (filePath) => {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${filePath}`);
    }
  };

  // Delete input and output files if they exist
  deleteFileIfExists(inputWavPath);
  deleteFileIfExists(outputWavPath);

  const pythonCommand = "python runTTS.py"; // Replace with your Python command

  const pythonProcess = exec(pythonCommand, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error}`);
      return;
    }
    console.log(`Python script output: ${stdout}`);
    console.error(`Python script errors: ${stderr}`);
  });

  pythonProcess.on("exit", (code) => {
    console.log(`Python script exited with code ${code}`);
    // Here, you can continue with any JavaScript code that depends on the Python script being completed.
    res.status(200).send("TTS Output Generated");
  });
});

var con = mysql.createConnection({
  host: "0.0.0.0",
  user: "root",
  password: "tanishq",
  database: "capstone",
});

con.connect(function (err) {
  if (err) throw err;
  console.log("Connected!");
});

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // Save uploaded files to the 'uploads' directory
  },
  filename: (req, file, cb) => {
    cb(null, "input.txt");
    console.log("Filename:", file.originalname);
  },
});

const upload = multer({ storage: storage });

const storageVideo = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // Save uploaded files to the 'uploads' directory
  },
  filename: (req, file, cb) => {
    cb(null, "inputVideo.mp4");
    console.log("Filename:", file.originalname);
  },
});

const uploadVideo = multer({ storage: storageVideo });

app.post("/uploadVideo", uploadVideo.single("file"), (req, res) => {
  // File has been uploaded and can be accessed as req.file
  console.log("Video file recieved");
  res.status(200).send("File uploaded successfully");
});

app.post("/uploadTextFile", upload.single("file"), (req, res) => {
  // File has been uploaded and can be accessed as req.file
  console.log("Text file recieved");
  res.status(200).send("Text File uploaded successfully");
});

app.post("/getgeneratedsummary", (req, res) => {
  fs.readFile(
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\outputs\\output.txt",
    "utf8",
    (err, data) => {
      if (err) {
        console.log(err);
        throw err;
      }
      res.status(200).send(data);
    }
  );
});

app.post("/updatesummary", (req, res) => {
  console.log("Update Summary recieved..");
  var obj = url.parse(req.url, true).query;
  console.log(obj.summary);

  var filePath =
    "C:\\Users\\Asus\\Desktop\\Capstone\\BackEnd\\outputs\\output.txt";
  var temp = obj.summary;

  //fs.writeFileSync(filePath, temp);

  fs.writeFile(filePath, temp, (err, data) => {
    if (err) {
      console.log(err);
      throw err;
    }
    res.status(200).send("Updated Summary");
  });

  //res.status(200).send();
});

app.post("/register", function (req, res) {
  console.log("Registration Received");

  console.log("Connected to Mysql");
  var obj = url.parse(req.url, true).query;
  console.log(obj);

  var newsqql = `SELECT * FROM users where username="${obj.user_name}"`;
  con.query(newsqql, function (errq, resq, fields) {
    if (errq) {
      throw errq;
    }
    if (resq.length != 0) {
      res.status(263).send("Username taken");
      return;
    }

    var sql = `INSERT INTO users VALUES ("${obj.name}","${obj.phn}","${obj.email}","${obj.gender}","${obj.age}","${obj.user_name}","${obj.pass}","NO")`;

    con.query(sql, function (err, res1) {
      if (err) {
        throw err;
      }
      console.log("1 record inserted in users table");
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.status(200).send("Registration Successful");
    });
  });
});

app.post("/updateConsent", (req, res) => {
  var obj = url.parse(req.url, true).query;
  console.log(obj);
  console.log("username is ");
  console.log(obj.username);

  var sql = `UPDATE users SET consent="YES" where username="${obj.username}"`;
  con.query(sql, function (err, res3) {
    if (err) {
      throw err;
    }
    res.status(200).send("Consent Updated !");
  });
});

app.post("/login", function (req, res) {
  console.log("Login Request Received...");

  var obj = url.parse(req.url, true).query;
  console.log(obj);
  console.log("user is ");
  console.log(obj.users);
  console.log("pass is ");
  console.log(obj.pass);

  var sql = `SELECT * from users WHERE username="${obj.users}"`;

  con.query(sql, function (err, res2, fields) {
    if (err) {
      throw err;
    }
    if (res2.length == 0) {
      res.status(201).send("Have not registered");
      return;
    }

    var found = 0;

    for (var i = 0; i < res2.length; i++) {
      if (res2[i].password == obj.pass) {
        var found = 1;
      }
    }

    if (found == 1) {
      var sql5 = `SELECT * FROM users WHERE username="${obj.users}"`;
      con.query(sql5, function (err8, res8, fields) {
        if (err8) {
          throw err8;
        }
        if (res8[0].consent == "NO") {
          console.log(234);
          res.status(234).send("Go to ConsentPage");
          return;
        } else {
          console.log(200);
          res.status(200).send("Go to dashboard");
          return;
        }
      });
    } else {
      res.status(208).send("Incorrect password");
      return;
    }
  });
});

console.log("server running at 9995..");
app.listen(9995);

const express = require('express')
const path = require('path')
const app = express()
const { exec } = require("child_process");
const port = 5006
var bodyParser = require('body-parser'); //connects bodyParsing middleware
var formidable = require('formidable');
var fs =require('fs-extra');    //File System-needed for renaming file etc
var file = require('fs');
var http = require('http');
const youtubedl = require('youtube-dl')

app.use(express.static(__dirname + '/'));
app.get('/', (req, res) => res.sendFile('index.html'))

app.route('/youtubedownload')
.post(function (req, res) {
  const video = youtubedl(req.body.name,
  // Optional arguments passed to youtube-dl.
  ['--format=18'],
  // Additional options can be given for calling `child_process.execFile()`.
  { cwd: __dirname })
 
// Will be called when the download starts.
video.on('info', function(info) {
  console.log('Download started')
  console.log('filename: ' + info._filename)
  console.log('size: ' + info.size)
})
 
video.pipe(fs.createWriteStream('rmdmy.wav'))
});


app.route('/')

.post(function (req, res) {
  req.setTimeout(0)
  var form = new formidable.IncomingForm();
    //Formidable uploads to operating systems tmp dir by default
    form.uploadDir = "./wavesurfer.js-master/example/annotation";       //set upload directory
    form.keepExtensions = true;     //keep file extension

    form.parse(req, function(err, fields, files){
        console.log("form.bytesReceived");
        //TESTING
        console.log("file size: "+JSON.stringify(files.fileUploaded.size));
        console.log("file path: "+JSON.stringify(files.fileUploaded.path));
        console.log("file name: "+JSON.stringify(files.fileUploaded.name));
        console.log("file type: "+JSON.stringify(files.fileUploaded.type));

        //Formidable changes the name of the uploaded file
        //Rename the file to its original name
        fs.rename(files.fileUploaded.path, './wavesurfer.js-master/example/annotation/rmdmy.wav');
        var child = require('child_process').exec('python3 demo/app.py')
	console.log('load annotation')
        child.stdout.pipe(process.stdout);
        child.stderr.pipe(process.stderr);
	child.on('exit', function() {
	    console.log("Process done");
	    res.redirect('/');
	    return;
        }) 
    });
});
var server = app.listen(80, function() {
console.log('Listening on port %d', server.address().port);
});
// var server = http.createServer(app);
// server.setTimeout(10*60*1000); // 10 * 60 seconds * 1000 msecs
// server.listen(3030, function () {
//     var logger = app.get('logger');
//     logger.info('**** STARTING SERVER ****');
// });

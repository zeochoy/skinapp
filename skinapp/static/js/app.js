$(document).ready(function(){
    console.log('test')
    var $status = $('.status');
    var results = {
        cat: [],
        prob: [],
        bprob: [],
        mprob: []
    };

    $('#img').change(function(event) {
        var obj = $(this)[0];
        // console.log(obj.files, obj.files[0])
        $status.html('');

        if (obj.files && obj.files[0]) {
            var fileReader = new FileReader();
            fileReader.onload = function(event) {
                $('.img-hidden').html(
                    `<img class="rounded border loaded-img shadow" id='loaded-img' src='${event.target.result}' height="300" width="300"/>`
                    );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                ctx.drawImage(img,0,0, 300, 300);
                });
            }
            fileReader.readAsDataURL(obj.files[0]);
        }
    });

    $('#predict').submit(function(event) {
        event.preventDefault();

        if ($('#img')[0].files.length === 0) {
            return false;
        }

        // remove active class
        $(".sample_img").removeClass("active-img");

        var imageData = new FormData($(this)[0]);
        console.log($(this)[0]);
        $status.html(
            `<span class='eval'>Evaluating...</span>`
        );

        $.ajax({
            url: '/predict',
            type: 'POST',
            processData: false,
            contentType: false,
            dataType: 'json',
            data: imageData,

            success: function(responseData) {
                if (responseData.error === 'bad-type') {
                    $status.html(
                        `<span class='eval'>Valid file types are .jpg and .png</span>`
                    );
                } else {
                    results["cat"] = responseData["cat"];
                    results["prob"] = responseData["prob"];
                    results["bprob"] = responseData["bprob"];
                    results["mprob"] = responseData["mprob"];
                    let cat = results["cat"]
                    let prob = results["prob"].toFixed(2)
                    let bprob = results["bprob"].toFixed(0)
                    let mprob = results["mprob"].toFixed(0)
                    $status.html(
                        `<span class='result success'>Results</span>
                         <span class='result-content'>${prob}% ${cat}</span>
                         <div class="progress">
                         <div class="progress-bar bg-success" role="progressbar" style="width: ${bprob}%" aria-valuenow="${bprob}" aria-valuemin="0" aria-valuemax="100"></div>
                         <div class="progress-bar bg-danger" role="progressbar" style="width: ${mprob}%" aria-valuenow="${mprob}" aria-valuemin="0" aria-valuemax="100"></div>
                         </div>`
                    );
                 }
            },
            error: function() {
                $status.html(
                    `<span class='eval failure'>Something went wrong, try again later.</span>`
                );
            }
        });
    });

    $(".sample_img").click(function() {

        // add active class to clicked picture
        $(".sample_img").removeClass("active-img");
        $(this).addClass("active-img");


        // grab image url
        var url = $(this).attr("src");

        // read url into blob using XHR
        var request = new XMLHttpRequest();
        request.open('GET', url, true);
        request.responseType = 'blob';

        request.onload = function() {
            var reader = new FileReader();
            reader.readAsDataURL(request.response);

            // draw canvas of selected sample image
            reader.onload =  function(e){
                // console.log('DataURL:', e.target.result);
                $('.img-hidden').html(
                    `<img class="rounded border loaded-img shadow" id='loaded-img' src='${e.target.result}' height="300" width="300"/>`
                    );
                //var c = document.getElementById("img-canvas");
                //var ctx = c.getContext("2d");
                //var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                    ctx.drawImage(img, 0,0, 300, 300);
                });

                // blob into form data
                var blob = request.response;
                var fd = new FormData();
                fd.set('file', blob);
                // console.log(formD);

                $status.html(
                    `<span class='eval'>Evaluating...</span>`
                );

                $.ajax({
                    url: '/predict',
                    type: 'POST',
                    processData: false,
                    contentType: false,
                    dataType: 'json',
                    data: fd,

                    success: function(responseData) {
                        if (responseData.error === 'bad-type') {
                            console.log('no good')
                            $status.html(
                                `<span class='eval'>Valid file types are .jpg and .png</span>`
                            );
                        } else {
                            results["cat"] = responseData["cat"];
                            results["prob"] = responseData["prob"];
                            results["bprob"] = responseData["bprob"];
                            results["mprob"] = responseData["mprob"];
                            let cat = results["cat"]
                            let prob = results["prob"].toFixed(2)
                            let bprob = results["bprob"].toFixed(0)
                            let mprob = results["mprob"].toFixed(0)
                            $status.html(
                                `<span class='result success'>Results</span>
                                 <span class='result-content'>${prob}% ${cat}</span>
                                 <div class="progress">
                                 <div class="progress-bar bg-success" role="progressbar" style="width: ${bprob}%" aria-valuenow="${bprob}" aria-valuemin="0" aria-valuemax="100"></div>
                                 <div class="progress-bar bg-danger" role="progressbar" style="width: ${mprob}%" aria-valuenow="${mprob}" aria-valuemin="0" aria-valuemax="100"></div>
                                 </div>`
                            );
                         }
                    },
                    error: function() {
                        $status.html(
                            `<span class='eval failure'>Something went wrong, try again later.</span>`
                        );
                    }
                });
            };
        };
        request.send();
    });
});

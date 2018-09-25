$(document).ready(function(){
    console.log('test')
    var $status = $('.status');
    var results = {
        cat: [],
        prob: []
    };

    $('#img').change(function(event) {
        var obj = $(this)[0];
        // console.log(obj.files, obj.files[0])
        $status.html('');

        if (obj.files && obj.files[0]) {
            var fileReader = new FileReader();
            fileReader.onload = function(event) {
                $('.img-hidden').html(
                    `<img id='loaded-img' src='${event.target.result}'/>`
                );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                ctx.drawImage(img,0,0, 500, 500);
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
                    let preData = JSON.stringify(responseData, null, '\t');
                    $status.html(
                        `<span class='result success'>Results</span>
                         <pre>${preData}</pre>`
                    );
                 }
            },
            error: function() {
                $status.html(
                    `<span class='eval'>Something went wrong, try again later.</span>`
                );
            }
        });
    });

    $(".sample_img").click(function() {

        // add active class to clicked picture
        $(".sample_img").removeClass("active");
        $(this).addClass("active");


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
                    `<img id='loaded-img' src='${e.target.result}'/>`
                    );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                    ctx.drawImage(img,0,0, 500, 500);
                });

                // blob into form data
                var blob = request.response;
                var fd = new FormData();
                fd.set('file', blob);
                // console.log(formD);

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
                            let preData = JSON.stringify(responseData, null, '\t');
                            $status.html(
                                `<span class='result success'>Results</span>
                                 <pre>${preData}</pre>`
                            );
                         }
                    },
                    error: function() {
                        $status.html(
                            `<span class='eval'>Something went wrong, try again later.</span>`
                        );
                    }
                });
            };
        };
        request.send();
    });
});

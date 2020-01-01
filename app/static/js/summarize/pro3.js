var address = 'http://localhost:9999';
// var address = 'http://49.234.19.31:8082';
// var address = 'http://111.229.161.111:8082';

$(function () {
    var result = {};

    var oPop = $('#pop_con');
    var iTime = 108;
    oPop.css({display: 'block'});
    var timer = setInterval(function () {
        iTime--;
        $('.progress-bar').css('width', 108 - iTime + '%');
        if (iTime == 0) {
            oPop.css({display: 'none'});
            clearInterval(timer);
            iTime = 108;
        }
    }, 80);

    $("#submit").click(function () {
        $('#firstpart_g').css('height', 200)
        $('#firstpart_g').css('padding', 23)
        $('#img').css("display", "block")
        $('#submit').css("display", "none")
        $('.pop_con').html('<div class="popup"><p id="pop_tip">抽奖中···</p><div class="progress"><div class="progress-bar" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100" style="width: 108%;"></div></div></div><div class="mask"></div>')
        var title = $(".form-control").val();
        var data = {'title': title}
        var req_json = JSON.stringify(data);
        console.log(data);
        $.ajax({
            url: address + '/choujiang/solve',
            type: 'post',
            data: req_json,
            contentType: "application/json",
            dataType: 'json',
        })
            .done(function (dat) {
                result = dat;
                $('#show').text(result['result']);
                $('.pop_con').html();
            })
    })
})





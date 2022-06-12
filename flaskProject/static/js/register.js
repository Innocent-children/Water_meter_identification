function bindCaptchaBtnClick() {
    $("#captcha-btn").on("click", function (event) {
        var $this = $(this);
        var email = $("input[name='email']").val();
        console.log(email);
        if (!email) {
            alert("请先输入邮箱！");
            return;
        } else if (email.indexOf("@") == -1) {
            alert("邮箱格式不正确！");
            return;
        }
        // 通过js发送网络请求：ajax。Async JavaScript And XML（JSON）
        // Jinja2
        $.ajax({
            url: "/user/captcha",
            method: "POST",
            data: {
                "email": email
            },
            success: function (res) {
                var code = res['code'];
                if (code == 200) {
                    // 取消点击事件
                    $this.off("click");
                    // 开始倒计时
                    var countDown = 60;
                    var timer = setInterval(function () {
                        countDown -= 1;
                        if (countDown > 0) {
                            $this.text(countDown + "秒后重新发送");
                        } else {
                            $this.text("获取验证码");
                            // 重新执行下这个函数，重新绑定点击事件
                            bindCaptchaBtnClick();
                            // 如果不需要倒计时了，那么就要记得清除倒计时，否则会一直执行下去
                            clearInterval(timer);
                        }
                    }, 1000);
                    alert("验证码发送成功！");
                } else {
                    alert(res['message']);
                }
            }
        })
    });
}


// 等网页文档所有元素都加载完成后再执行
$(function () {
    bindCaptchaBtnClick();
});
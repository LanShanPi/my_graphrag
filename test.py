

# from openai import OpenAI
# client = OpenAI(  
#     api_key=OPENAI_API_KEY1, 
#     base_url=API_BASE1
#      )

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     # messages=[
#     #     {"role": "system", "content": "根据你对中国历史古籍的理解，回答问题"},
#     #     {"role": "user","content": "王熙凤有哪些别名，显示格式为：&名字1&名字2...名字n&"}
#     # ]
#     messages=[
#         {"role": "system", "content": "根据你对中国历史古籍的理解，回答问题"},
#         {"role": "user","content": "背一首满江红"}
#     ]
# )
# print(completion.choices[0].message.content)
# # print(completion.choices[0].message.content.split("&")[1:-1])


arg1 = {
      "cost_time": 13,
      "count": 2,
      "data": [
        {
          "Goods_Title": "【中信童书】狐狸家三国演义+西游记套装 全25册 赠故事机帆布包1-8",
          "ShopName": "中信出版童书旗舰店",
          "ImageUrl": "https://img3.feigua.cn/img/ecom-shop-material/png_m_dcd3a85d0a2dd20f6abb12740a551ecd_sx_2596492_www1417-1417~tplv-resize:800:800.webp?$$dyurl=https://p3-aio.ecombdimg.com/img/ecom-shop-material/png_m_dcd3a85d0a2dd20f6abb12740a551ecd_sx_2596492_www1417-1417~tplv-resize:800:800.webp",
          "Zhibo_info": [
            {
              "zhibo_Title": "洗脸巾心相印一次性纯棉洗脸巾珍珠纹棉柔巾全棉时代羽绒服湿巾德佑卸妆湿巾",
              "zhibo_CoverUrl": "https://p11-webcast-sign.douyinpic.com/webcast-cover/7452317962477865779~tplv-qz53dukwul-common-resize:0:0.image?biz_tag=aweme_webcast&from=webcast.room.pack&l=20250107235630700826B9776241304893&lk3s=39e7556e&s=reflow_room_info&sc=webcast_cover&x-expires=1738857390&x-signature=CAFczWnbfmi29DZn5qWaYEcr4GQ%3D&$$dyurl=https://logocdn.feigua.cn/RoomLogo/202412/272579faa38c67b968aa9dcedf1fb2f6.jpg-jpg"
            },
            {
              "zhibo_Title": "2025年，雯潞与你一起共同精进提升自己！",
              "zhibo_CoverUrl": "https://p3-webcast.douyinpic.com/img/aweme-avatar/tos-cn-avt-0015_6a717727a72712c70e9768c9096fa61e~tplv-resize:0:0.image?biz_tag=aweme_webcast&from=webcast.room.pack&l=20250108124056489EA0A3E1B10309A34E&s=reflow_room_info&sc=webcast_cover&$$dyurl=https://logocdn.feigua.cn/RoomLogo/202501/d7367585590ab3c72ec06c7a04595039.jpg-jpg"
            }
          ],
          "Video_info": [
            {
              "video_Title": "洗脸巾这种消耗品！这个价！刷到就囤了吧！#洗脸巾 #一次性洗脸巾 #平价好物 #实用好物",
              "video_CoverUrl": "https://p11-sign.douyinpic.com/tos-cn-p-0015c000-ce/oYPEnnwAEBCrMeycvpiJwTIieA6SBievA6CzGg~tplv-dy-resize-walign-adapt-aq:540:q75.jpeg?lk3s=138a59ce&x-expires=1735322400&x-signature=olcqH8RQ1GLi5FjcqosDsj%2Fm2Tk%3D&from=327834062&s=PackSourceEnum_PUBLISH&se=false&sc=cover&biz_tag=aweme_video&l=20241214021211F976A3D69E33BE22E6B9&$$dyurl=https://logocdn.feigua.cn/Video/202412/0672d5267370c076a969f34f4cd82f66.jpg-jpg"
            },
            {
              "video_Title": "洗脸巾这种消耗品！这个价！刷到就囤了吧！#洗脸巾 #一次性洗脸巾 #平价好物 #实用好物",
              "video_CoverUrl": "https://p9-sign.douyinpic.com/tos-cn-p-0015/ooPuGej3BItEIsCgMDOALCfcEQ7BAkCKpBCvfC~tplv-dy-resize-walign-adapt-aq:540:q75.jpeg?lk3s=138a59ce&x-expires=1735855200&x-signature=pL3aCVL1a9USEkCFxOXuaP3GN9I%3D&from=327834062&s=PackSourceEnum_PUBLISH&se=false&sc=cover&biz_tag=aweme_video&l=202412200609443C1246FDB3A7741D2BB5&$$dyurl=https://logocdn.feigua.cn/Video/202412/1f219acb94ca1e3982fa7fe48d41344e.jpg-jpg"
            }
          ],
          "Image_info": {
            "Gender": [
              {
                "男性": "25.50%"
              },
              {
                "女性": "74.50%"
              }
            ],
            "Age": [
              {
                "18-23": "16.40%"
              },
              {
                "24-30": "24.78%"
              },
              {
                "31-40": "30.88%"
              },
              {
                "41-50": "15.44%"
              },
              {
                "50+": "12.50%"
              }
            ],
            "Hobbys": [
              {
                "母婴": "9.95%"
              },
              {
                "运动": "9.95%"
              },
              {
                "萌宠": "8.77%"
              },
              {
                "游戏": "7.82%"
              },
              {
                "餐饮美食": "6.87%"
              },
              {
                "教育": "6.87%"
              },
              {
                "穿搭": "6.16%"
              },
              {
                "文艺": "5.45%"
              },
              {
                "美妆": "5.45%"
              },
              {
                "影视娱乐": "5.45%"
              }
            ]
          }
        },
        {
          "Goods_Title": "【送礼佳选四大名著四册礼盒】红楼梦西游记水浒传三国演义立体书",
          "ShopName": "青葫芦立体书官方旗舰店",
          "ImageUrl": "https://img3.feigua.cn/obj/ecom-shop-material/png_m_bef17f1a530b75e358923226c8275a85_sx_1249305_www800-800?$$dyurl=https://p3-aio.ecombdimg.com/obj/ecom-shop-material/png_m_bef17f1a530b75e358923226c8275a85_sx_1249305_www800-800",
          "Zhibo_info": [
            {
              "zhibo_Title": "洗脸巾心相印一次性纯棉洗脸巾珍珠纹棉柔巾全棉时代羽绒服湿巾德佑卸妆湿巾",
              "zhibo_CoverUrl": "https://p11-webcast-sign.douyinpic.com/webcast-cover/7452317962477865779~tplv-qz53dukwul-common-resize:0:0.image?biz_tag=aweme_webcast&from=webcast.room.pack&l=20250107235630700826B9776241304893&lk3s=39e7556e&s=reflow_room_info&sc=webcast_cover&x-expires=1738857390&x-signature=CAFczWnbfmi29DZn5qWaYEcr4GQ%3D&$$dyurl=https://logocdn.feigua.cn/RoomLogo/202412/272579faa38c67b968aa9dcedf1fb2f6.jpg-jpg"
            },
            {
              "zhibo_Title": "2025年，雯潞与你一起共同精进提升自己！",
              "zhibo_CoverUrl": "https://p3-webcast.douyinpic.com/img/aweme-avatar/tos-cn-avt-0015_6a717727a72712c70e9768c9096fa61e~tplv-resize:0:0.image?biz_tag=aweme_webcast&from=webcast.room.pack&l=20250108124056489EA0A3E1B10309A34E&s=reflow_room_info&sc=webcast_cover&$$dyurl=https://logocdn.feigua.cn/RoomLogo/202501/d7367585590ab3c72ec06c7a04595039.jpg-jpg"
            }
          ],
          "Video_info": [
            {
              "video_Title": "洗脸巾这种消耗品！这个价！刷到就囤了吧！#洗脸巾 #一次性洗脸巾 #平价好物 #实用好物",
              "video_CoverUrl": "https://p11-sign.douyinpic.com/tos-cn-p-0015c000-ce/oYPEnnwAEBCrMeycvpiJwTIieA6SBievA6CzGg~tplv-dy-resize-walign-adapt-aq:540:q75.jpeg?lk3s=138a59ce&x-expires=1735322400&x-signature=olcqH8RQ1GLi5FjcqosDsj%2Fm2Tk%3D&from=327834062&s=PackSourceEnum_PUBLISH&se=false&sc=cover&biz_tag=aweme_video&l=20241214021211F976A3D69E33BE22E6B9&$$dyurl=https://logocdn.feigua.cn/Video/202412/0672d5267370c076a969f34f4cd82f66.jpg-jpg"
            },
            {
              "video_Title": "洗脸巾这种消耗品！这个价！刷到就囤了吧！#洗脸巾 #一次性洗脸巾 #平价好物 #实用好物",
              "video_CoverUrl": "https://p9-sign.douyinpic.com/tos-cn-p-0015/ooPuGej3BItEIsCgMDOALCfcEQ7BAkCKpBCvfC~tplv-dy-resize-walign-adapt-aq:540:q75.jpeg?lk3s=138a59ce&x-expires=1735855200&x-signature=pL3aCVL1a9USEkCFxOXuaP3GN9I%3D&from=327834062&s=PackSourceEnum_PUBLISH&se=false&sc=cover&biz_tag=aweme_video&l=202412200609443C1246FDB3A7741D2BB5&$$dyurl=https://logocdn.feigua.cn/Video/202412/1f219acb94ca1e3982fa7fe48d41344e.jpg-jpg"
            }
          ],
          "Image_info": {
            "Gender": [
              {
                "男性": "25.50%"
              },
              {
                "女性": "74.50%"
              }
            ],
            "Age": [
              {
                "18-23": "16.40%"
              },
              {
                "24-30": "24.78%"
              },
              {
                "31-40": "30.88%"
              },
              {
                "41-50": "15.44%"
              },
              {
                "50+": "12.50%"
              }
            ],
            "Hobbys": [
              {
                "母婴": "9.95%"
              },
              {
                "运动": "9.95%"
              },
              {
                "萌宠": "8.77%"
              },
              {
                "游戏": "7.82%"
              },
              {
                "餐饮美食": "6.87%"
              },
              {
                "教育": "6.87%"
              },
              {
                "穿搭": "6.16%"
              },
              {
                "文艺": "5.45%"
              },
              {
                "美妆": "5.45%"
              },
              {
                "影视娱乐": "5.45%"
              }
            ]
          }
        }
      ]
    }



import json
def main(arg1) -> dict:
    result = []
    if arg1:
        for item in arg1[0]["data"][0]["Image_info"]["Age"]:
            result.append(list(item.keys())[0]+":"+item[list(item.keys())[0]])
        for item in arg1[0]["data"][0]["Image_info"]["Gender"]:
            result.append(list(item.keys())[0]+":"+item[list(item.keys())[0]])
    return {
        "result":result
    }
print(main(arg1))
from magic_pdf.libs.commons import fitz             # pyMuPDF库


def parse_headers(page_ID: int, page: fitz.Page, json_from_DocXchain_obj: dict):
    """
    :param page_ID: int类型，当前page在当前pdf文档中是第page_D页。
    :param page :fitz读取的当前页的内容
    :param res_dir_path: str类型，是每一个pdf文档，在当前.py文件的目录下生成一个与pdf文档同名的文件夹，res_dir_path就是文件夹的dir
    :param json_from_DocXchain_obj: dict类型，把pdf文档送入DocXChain模型中后，提取bbox，结果保存到pdf文档同名文件夹下的 page_ID.json文件中了。json_from_DocXchain_obj就是打开后的dict
    """
    DPI = 72  # use this resolution
    pix = page.get_pixmap(dpi=DPI)
    pageL = 0
    pageR = int(pix.w)
    pageU = 0
    pageD = int(pix.h)
    

    #--------- 通过json_from_DocXchain来获取 header ---------#
    header_bbox_from_DocXChain = []

    xf_json = json_from_DocXchain_obj
    width_from_json = xf_json['page_info']['width']
    height_from_json = xf_json['page_info']['height']
    LR_scaleRatio = width_from_json / (pageR - pageL)
    UD_scaleRatio = height_from_json / (pageD - pageU)

    # {0: 'title',  # 标题
    # 1: 'figure', # 图片
    #  2: 'plain text',  # 文本
    #  3: 'header',      # 页眉
    #  4: 'page number', # 页码
    #  5: 'footnote',    # 脚注
    #  6: 'footer',      # 页脚
    #  7: 'table',       # 表格
    #  8: 'table caption',  # 表格描述
    #  9: 'figure caption', # 图片描述
    #  10: 'equation',      # 公式
    #  11: 'full column',   # 单栏
    #  12: 'sub column',    # 多栏
    #  13: 'embedding',     # 嵌入公式
    #  14: 'isolated'}      # 单行公式
    for xf in xf_json['layout_dets']:
        L = xf['poly'][0] / LR_scaleRatio
        U = xf['poly'][1] / UD_scaleRatio
        R = xf['poly'][2] / LR_scaleRatio
        D = xf['poly'][5] / UD_scaleRatio
        # L += pageL          # 有的页面，artBox偏移了。不在（0,0）
        # R += pageL
        # U += pageU
        # D += pageU
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        if xf['category_id'] == 3 and xf['score'] >= 0.3:
            header_bbox_from_DocXChain.append((L, U, R, D))
            
    
    header_final_names = []
    header_final_bboxs = []
    header_ID = 0
    for L, U, R, D in header_bbox_from_DocXChain:
        # cur_header = page.get_pixmap(clip=(L,U,R,D))
        new_header_name = "header_{}_{}.png".format(page_ID, header_ID)    # 页眉name
        # cur_header.save(res_dir_path + '/' + new_header_name)           # 把页眉存储在新建的文件夹，并命名
        header_final_names.append(new_header_name)                        # 把页面的名字存在list中
        header_final_bboxs.append((L, U, R, D))
        header_ID += 1
        

    header_final_bboxs.sort(key = lambda LURD: (LURD[1], LURD[0]))
    curPage_all_header_bboxs = header_final_bboxs
    return curPage_all_header_bboxs


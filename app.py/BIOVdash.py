import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np
import base64
from dash import no_update
from io import BytesIO as _BytesIO

#from pymed import PubMed


def layers(howmany, resolution, rim1, rim2, rim3, rim4):
    # values
    color = ["blue", "red", "green"]
    number_of_pixels = int(resolution)

    if howmany == 1:
        # obtain pixel intensity in a nested list for image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        z1 = (np.sum(np.asarray(image11, dtype="int32"), axis=2))

        # give figure data
        fig = go.Figure(data=[
            go.Surface(z=z1, colorscale=color, showscale=False)])

        return fig

    elif howmany == 2:
        # obtain pixel intensity in a nested list for image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        z1 = (np.sum(np.asarray(image11, dtype="int32"), axis=2))

        # obtain pixel intensity in a nested list for image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        z2 = (np.sum(np.asarray(image22, dtype="int32"), axis=2)) + (np.amax(z1) + 2000)

        # give figure data
        fig = go.Figure(data=[
            go.Surface(z=z1, colorscale=color, showscale=False),
            go.Surface(z=z2, colorscale=color, showscale=False)])

        return fig

    elif howmany == 3:
        # obtain pixel intensity in a nested list for image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        z1 = (np.sum(np.asarray(image11, dtype="int32"), axis=2))

        # obtain pixel intensity in a nested list for image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        z2 = (np.sum(np.asarray(image22, dtype="int32"), axis=2)) + (np.amax(z1) + 2000)

        # obtain pixel intensity in a nested list for image 3
        new_image33 = rim3.resize((number_of_pixels, number_of_pixels))
        image33 = new_image33.convert('RGB')

        z3 = (np.sum(np.asarray(image33, dtype="int32"), axis=2)) + (np.amax(z2) + 2000)

        # give figure data
        fig = go.Figure(data=[
            go.Surface(z=z1, colorscale=color, showscale=False),
            go.Surface(z=z2, colorscale=color, showscale=False),
            go.Surface(z=z3, colorscale=color, showscale=False)])

        return fig

    elif howmany == 4:
        # obtain pixel intensity in a nested list for image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        z1 = (np.sum(np.asarray(image11, dtype="int32"), axis=2))

        # obtain pixel intensity in a nested list for image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        z2 = (np.sum(np.asarray(image22, dtype="int32"), axis=2)) + (np.amax(z1) + 2000)

        # obtain pixel intensity in a nested list for image 3
        new_image33 = rim3.resize((number_of_pixels, number_of_pixels))
        image33 = new_image33.convert('RGB')

        z3 = (np.sum(np.asarray(image33, dtype="int32"), axis=2)) + (np.amax(z2) + 2000)

        # obtain pixel intensity in a nested list for image 4
        new_image44 = rim4.resize((number_of_pixels, number_of_pixels))
        image44 = new_image44.convert('RGB')

        z4 = (np.sum(np.asarray(image44, dtype="int32"), axis=2)) + (np.amax(z3) + 2000)

        # give figure data
        fig = go.Figure(data=[
            go.Surface(z=z1, colorscale=color, showscale=False),
            go.Surface(z=z2, colorscale=color, showscale=False),
            go.Surface(z=z3, colorscale=color, showscale=False),
            go.Surface(z=z4, colorscale=color, showscale=False)])

        #username = ruser
        #api_key = rapi
        #chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        #return py.plot(fig, filename='en1', auto_open=False)
        return fig


def colored_layers(howmany, resolution, rim1, rim2, rim3, rim4):
    # define variables
    pixels_per_box = 16
    number_of_boxes = int(int(resolution)/16)
    total_pixels = pixels_per_box * number_of_boxes

    # make x
    x = [1, 1]

    for i in range(2, (pixels_per_box + 2)):
        x.append(i - 0.5)
        for e in range(3):
            x.append(i)
    del x[len(x) - 1]

    if howmany == 1:
        # resize image 1 to the desired dimensions
        new_image11 = rim1.resize((total_pixels, total_pixels))
        image11 = new_image11.convert('RGB')

        # obtain the rgb value and intensity of each pixel in images
        rgb_list1 = []

        for rgb_b in range(number_of_boxes):
            for rgb_a in range(number_of_boxes):
                for col in range(pixels_per_box):

                    for row in range(pixels_per_box):
                        red1, green1, blue1 = image11.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))

                        rgb1 = red1, green1, blue1

                        rgb_list1.append(rgb1)
        # make color scale for images
        w = 1 / ((pixels_per_box ** 2) - 1)
        color_scale1 = []
        var = 0

        for i in range(number_of_boxes ** 2):
            lcv = var
            color_scale1.append([])

            for e in range(pixels_per_box ** 2):
                var1 = (w * e), px.colors.label_rgb((rgb_list1[lcv + i + e]))

                ring1 = list(var1)

                color_scale1[i].append(ring1)

                var = lcv + e

        # make z1
        z1sums = np.sum(np.asarray(image11, dtype="int32"), axis=2)
        z1reshape1 = np.reshape(z1sums,
                                ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z1transpose1 = np.transpose(z1reshape1, (0, 2, 1))
        z1reshape2 = np.reshape(z1transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z1transpose2 = np.transpose(z1reshape2, (0, 1, 2))
        z1repeat = np.repeat(np.repeat(z1transpose2, 3, axis=1), 3, axis=2)
        z1 = np.insert(np.insert(z1repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1)

        surfarray = np.arange(1 / 256, 1 + 1 / 256, 1 / 256)
        surfreshape = np.reshape(surfarray, (16, 16))
        surfrepeat = np.repeat(np.repeat(surfreshape, 3, axis=0), 3, axis=1)
        surfquick = np.insert(np.insert(surfrepeat, np.arange(0, 49, 3), 0, axis=1), np.arange(0, 49, 3), 0,
                              axis=0)

        # provide data for plot
        data = []
        var = 0

        for i in range(number_of_boxes):
            lcv = var
            for e in range(number_of_boxes):
                ax1 = np.asarray([n + ((pixels_per_box - 1) * i) for n in x])

                ay1 = np.asarray([o + ((pixels_per_box - 1) * e) for o in x])

                asurf1 = surfquick

                datum1 = go.Surface(z=z1[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale1[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                var = lcv + e
                data.append(datum1)

        # setup and display figure
        fig = go.Figure(data=data)
        return fig

    if howmany == 2:
        # resize image 1 to the desired dimensions
        new_image11 = rim1.resize((total_pixels, total_pixels))
        image11 = new_image11.convert('RGB')

        # resize image 2 to the desired dimensions
        new_image22 = rim2.resize((total_pixels, total_pixels))
        image22 = new_image22.convert('RGB')

        # obtain the rgb value and intensity of each pixel in images
        rgb_list1 = []
        rgb_list2 = []

        for rgb_b in range(number_of_boxes):
            for rgb_a in range(number_of_boxes):
                for col in range(pixels_per_box):

                    for row in range(pixels_per_box):
                        red1, green1, blue1 = image11.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red2, green2, blue2 = image22.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))

                        rgb1 = red1, green1, blue1
                        rgb2 = red2, green2, blue2

                        rgb_list1.append(rgb1)
                        rgb_list2.append(rgb2)

        # make color scale for images
        w = 1 / ((pixels_per_box ** 2) - 1)
        color_scale1 = []
        color_scale2 = []
        var = 0

        for i in range(number_of_boxes ** 2):
            lcv = var
            color_scale1.append([])
            color_scale2.append([])

            for e in range(pixels_per_box ** 2):
                var1 = (w * e), px.colors.label_rgb((rgb_list1[lcv + i + e]))
                var2 = (w * e), px.colors.label_rgb((rgb_list2[lcv + i + e]))

                ring1 = list(var1)
                ring2 = list(var2)

                color_scale1[i].append(ring1)
                color_scale2[i].append(ring2)

                var = lcv + e

        # make z1
        z1sums = np.sum(np.asarray(image11, dtype="int32"), axis=2)
        z1reshape1 = np.reshape(z1sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z1transpose1 = np.transpose(z1reshape1, (0, 2, 1))
        z1reshape2 = np.reshape(z1transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z1transpose2 = np.transpose(z1reshape2, (0, 1, 2))
        z1repeat = np.repeat(np.repeat(z1transpose2, 3, axis=1), 3, axis=2)
        z1 = np.insert(np.insert(z1repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1)

        # make z2
        z2sums = np.sum(np.asarray(image22, dtype="int32"), axis=2)
        z2reshape1 = np.reshape(z2sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z2transpose1 = np.transpose(z2reshape1, (0, 2, 1))
        z2reshape2 = np.reshape(z2transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z2transpose2 = np.transpose(z2reshape2, (0, 1, 2))
        z2repeat = np.repeat(np.repeat(z2transpose2, 3, axis=1), 3, axis=2)
        z2 = np.insert(np.insert(z2repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z1) + 1000

        surfarray = np.arange(1 / 256, 1 + 1 / 256, 1 / 256)
        surfreshape = np.reshape(surfarray, (16, 16))
        surfrepeat = np.repeat(np.repeat(surfreshape, 3, axis=0), 3, axis=1)
        surfquick = np.insert(np.insert(surfrepeat, np.arange(0, 49, 3), 0, axis=1), np.arange(0, 49, 3), 0, axis=0)

        # provide data for plot
        data = []
        var = 0

        for i in range(number_of_boxes):
            lcv = var
            for e in range(number_of_boxes):
                ax1 = np.asarray([n + ((pixels_per_box - 1) * i) for n in x])

                ay1 = np.asarray([o + ((pixels_per_box - 1) * e) for o in x])

                asurf1 = surfquick

                datum1 = go.Surface(z=z1[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale1[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum2 = go.Surface(z=z2[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale2[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                var = lcv + e
                data.append(datum1)
                data.append(datum2)

        # setup and display figure
        fig = go.Figure(data=data)
        return fig

    if howmany == 3:
        # resize image 1 to the desired dimensions
        new_image11 = rim1.resize((total_pixels, total_pixels))
        image11 = new_image11.convert('RGB')

        # resize image 2 to the desired dimensions
        new_image22 = rim2.resize((total_pixels, total_pixels))
        image22 = new_image22.convert('RGB')

        # resize image 3 to the desired dimensions
        new_image33 = rim3.resize((total_pixels, total_pixels))
        image33 = new_image33.convert('RGB')

        # obtain the rgb value and intensity of each pixel in images
        rgb_list1 = []
        rgb_list2 = []
        rgb_list3 = []

        for rgb_b in range(number_of_boxes):
            for rgb_a in range(number_of_boxes):
                for col in range(pixels_per_box):

                    for row in range(pixels_per_box):
                        red1, green1, blue1 = image11.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red2, green2, blue2 = image22.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red3, green3, blue3 = image33.getpixel(
                            ((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))

                        rgb1 = red1, green1, blue1
                        rgb2 = red2, green2, blue2
                        rgb3 = red3, green3, blue3

                        rgb_list1.append(rgb1)
                        rgb_list2.append(rgb2)
                        rgb_list3.append(rgb3)

        # make color scale for images
        w = 1 / ((pixels_per_box ** 2) - 1)
        color_scale1 = []
        color_scale2 = []
        color_scale3 = []
        var = 0

        for i in range(number_of_boxes ** 2):
            lcv = var
            color_scale1.append([])
            color_scale2.append([])
            color_scale3.append([])

            for e in range(pixels_per_box ** 2):
                var1 = (w * e), px.colors.label_rgb((rgb_list1[lcv + i + e]))
                var2 = (w * e), px.colors.label_rgb((rgb_list2[lcv + i + e]))
                var3 = (w * e), px.colors.label_rgb((rgb_list3[lcv + i + e]))

                ring1 = list(var1)
                ring2 = list(var2)
                ring3 = list(var3)

                color_scale1[i].append(ring1)
                color_scale2[i].append(ring2)
                color_scale3[i].append(ring3)

                var = lcv + e

        # make z1
        z1sums = np.sum(np.asarray(image11, dtype="int32"), axis=2)
        z1reshape1 = np.reshape(z1sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z1transpose1 = np.transpose(z1reshape1, (0, 2, 1))
        z1reshape2 = np.reshape(z1transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z1transpose2 = np.transpose(z1reshape2, (0, 1, 2))
        z1repeat = np.repeat(np.repeat(z1transpose2, 3, axis=1), 3, axis=2)
        z1 = np.insert(np.insert(z1repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1)

        # make z2
        z2sums = np.sum(np.asarray(image22, dtype="int32"), axis=2)
        z2reshape1 = np.reshape(z2sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z2transpose1 = np.transpose(z2reshape1, (0, 2, 1))
        z2reshape2 = np.reshape(z2transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z2transpose2 = np.transpose(z2reshape2, (0, 1, 2))
        z2repeat = np.repeat(np.repeat(z2transpose2, 3, axis=1), 3, axis=2)
        z2 = np.insert(np.insert(z2repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z1) + 1000

        # make z3
        z3sums = np.sum(np.asarray(image33, dtype="int32"), axis=2)
        z3reshape1 = np.reshape(z3sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z3transpose1 = np.transpose(z3reshape1, (0, 2, 1))
        z3reshape2 = np.reshape(z3transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z3transpose2 = np.transpose(z3reshape2, (0, 1, 2))
        z3repeat = np.repeat(np.repeat(z3transpose2, 3, axis=1), 3, axis=2)
        z3 = np.insert(np.insert(z3repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z2) + 1000

        surfarray = np.arange(1 / 256, 1 + 1 / 256, 1 / 256)
        surfreshape = np.reshape(surfarray, (16, 16))
        surfrepeat = np.repeat(np.repeat(surfreshape, 3, axis=0), 3, axis=1)
        surfquick = np.insert(np.insert(surfrepeat, np.arange(0, 49, 3), 0, axis=1), np.arange(0, 49, 3), 0, axis=0)

        # provide data for plot
        data = []
        var = 0

        for i in range(number_of_boxes):
            lcv = var
            for e in range(number_of_boxes):
                ax1 = np.asarray([n + ((pixels_per_box - 1) * i) for n in x])

                ay1 = np.asarray([o + ((pixels_per_box - 1) * e) for o in x])

                asurf1 = surfquick

                datum1 = go.Surface(z=z1[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale1[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum2 = go.Surface(z=z2[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale2[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum3 = go.Surface(z=z3[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale3[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                var = lcv + e
                data.append(datum1)
                data.append(datum2)
                data.append(datum3)

        # setup and display figure
        fig = go.Figure(data=data)
        return fig

    if howmany == 4:
        # resize image 1 to the desired dimensions
        new_image11 = rim1.resize((total_pixels, total_pixels))
        image11 = new_image11.convert('RGB')

        # resize image 2 to the desired dimensions
        new_image22 = rim2.resize((total_pixels, total_pixels))
        image22 = new_image22.convert('RGB')

        # resize image 3 to the desired dimensions
        new_image33 = rim3.resize((total_pixels, total_pixels))
        image33 = new_image33.convert('RGB')
        # resize image 4 to the desired dimensions
        new_image44 = rim4.resize((total_pixels, total_pixels))
        image44 = new_image44.convert('RGB')

        # obtain the rgb value and intensity of each pixel in images
        rgb_list1 = []
        rgb_list2 = []
        rgb_list3 = []
        rgb_list4 = []

        for rgb_b in range(number_of_boxes):
            for rgb_a in range(number_of_boxes):
                for col in range(pixels_per_box):

                    for row in range(pixels_per_box):
                        red1, green1, blue1 = image11.getpixel(((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red2, green2, blue2 = image22.getpixel(((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red3, green3, blue3 = image33.getpixel(((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))
                        red4, green4, blue4 = image44.getpixel(((col + (pixels_per_box * rgb_a)), (row + (pixels_per_box * rgb_b))))

                        rgb1 = red1, green1, blue1
                        rgb2 = red2, green2, blue2
                        rgb3 = red3, green3, blue3
                        rgb4 = red4, green4, blue4

                        rgb_list1.append(rgb1)
                        rgb_list2.append(rgb2)
                        rgb_list3.append(rgb3)
                        rgb_list4.append(rgb4)

        # make color scale for images
        w = 1 / ((pixels_per_box ** 2) - 1)
        color_scale1 = []
        color_scale2 = []
        color_scale3 = []
        color_scale4 = []
        var = 0

        for i in range(number_of_boxes ** 2):
            lcv = var
            color_scale1.append([])
            color_scale2.append([])
            color_scale3.append([])
            color_scale4.append([])

            for e in range(pixels_per_box ** 2):
                var1 = (w * e), px.colors.label_rgb((rgb_list1[lcv + i + e]))
                var2 = (w * e), px.colors.label_rgb((rgb_list2[lcv + i + e]))
                var3 = (w * e), px.colors.label_rgb((rgb_list3[lcv + i + e]))
                var4 = (w * e), px.colors.label_rgb((rgb_list4[lcv + i + e]))

                ring1 = list(var1)
                ring2 = list(var2)
                ring3 = list(var3)
                ring4 = list(var4)

                color_scale1[i].append(ring1)
                color_scale2[i].append(ring2)
                color_scale3[i].append(ring3)
                color_scale4[i].append(ring4)

                var = lcv + e

        # make z1
        z1sums = np.sum(np.asarray(image11, dtype="int32"), axis=2)
        z1reshape1 = np.reshape(z1sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z1transpose1 = np.transpose(z1reshape1, (0, 2, 1))
        z1reshape2 = np.reshape(z1transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z1transpose2 = np.transpose(z1reshape2, (0, 1, 2))
        z1repeat = np.repeat(np.repeat(z1transpose2, 3, axis=1), 3, axis=2)
        z1 = np.insert(np.insert(z1repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1)

        # make z2
        z2sums = np.sum(np.asarray(image22, dtype="int32"), axis=2)
        z2reshape1 = np.reshape(z2sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z2transpose1 = np.transpose(z2reshape1, (0, 2, 1))
        z2reshape2 = np.reshape(z2transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z2transpose2 = np.transpose(z2reshape2, (0, 1, 2))
        z2repeat = np.repeat(np.repeat(z2transpose2, 3, axis=1), 3, axis=2)
        z2 = np.insert(np.insert(z2repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z1) + 1000

        # make z3
        z3sums = np.sum(np.asarray(image33, dtype="int32"), axis=2)
        z3reshape1 = np.reshape(z3sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z3transpose1 = np.transpose(z3reshape1, (0, 2, 1))
        z3reshape2 = np.reshape(z3transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z3transpose2 = np.transpose(z3reshape2, (0, 1, 2))
        z3repeat = np.repeat(np.repeat(z3transpose2, 3, axis=1), 3, axis=2)
        z3 = np.insert(np.insert(z3repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z2) + 1000

        # make z4
        z4sums = np.sum(np.asarray(image44, dtype="int32"), axis=2)
        z4reshape1 = np.reshape(z4sums, ((int((int(resolution) ** 2) / (int(resolution) * 16))), 16, int(resolution)))
        z4transpose1 = np.transpose(z4reshape1, (0, 2, 1))
        z4reshape2 = np.reshape(z4transpose1, ((int((int(resolution) ** 2) / 256)), 16, 16))

        z4transpose2 = np.transpose(z4reshape2, (0, 1, 2))
        z4repeat = np.repeat(np.repeat(z4transpose2, 3, axis=1), 3, axis=2)
        z4 = np.insert(np.insert(z4repeat, np.arange(0, 48, 3), 0, axis=2), np.arange(0, 48, 3), 0, axis=1) \
             + np.amax(z3) + 1000

        surfarray = np.arange(1/256, 1+1/256, 1/256)
        surfreshape = np.reshape(surfarray, (16, 16))
        surfrepeat = np.repeat(np.repeat(surfreshape, 3, axis=0), 3, axis=1)
        surfquick = np.insert(np.insert(surfrepeat, np.arange(0, 49, 3), 0, axis=1), np.arange(0, 49, 3), 0, axis=0)

        # provide data for plot
        data = []
        var = 0

        for i in range(number_of_boxes):
            lcv = var
            for e in range(number_of_boxes):

                ax1 = np.asarray([n + ((pixels_per_box - 1) * i) for n in x])

                ay1 = np.asarray([o + ((pixels_per_box - 1) * e) for o in x])

                asurf1 = surfquick

                datum1 = go.Surface(z=z1[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale1[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum2 = go.Surface(z=z2[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale2[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum3 = go.Surface(z=z3[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale3[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                datum4 = go.Surface(z=z4[lcv + i + e], x=ax1,
                                    y=ay1,
                                    colorscale=color_scale4[lcv + i + e],
                                    surfacecolor=asurf1,
                                    connectgaps=False)

                var = lcv + e
                data.append(datum1)
                data.append(datum2)
                data.append(datum3)
                data.append(datum4)

        # setup and display figure
        fig = go.Figure(data=data)
        #pio.write_html(fig, file='www/colored_layers.html', auto_open=False)
        #username = ruser
        #api_key = rapi
        #chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        #return py.plot(fig, filename='en2', auto_open=False)
        return fig

def open_image(resolution, imm):
    img = imm
    fast_x = []
    fast_y = []
    fast_z = []

    for i in range(1, 74):

        img.seek(i)

        img1 = img.resize((int(resolution), int(resolution)))
        rgb_im = img1.convert('RGB')
        data = np.insert(np.transpose(np.nonzero(np.sum(np.asarray(rgb_im, dtype="int32"), axis=2))), 2, i, axis=1)

        if data is not []:
            for n in data:
                fast_x.append(n[0])
                fast_y.append(n[1])
                fast_z.append(n[2])

    return fast_x, fast_y, fast_z


def dots(resolution, imm):

    # open image
    x, y, z = open_image(resolution, imm)

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers', marker=dict(
                                           size=0.8, symbol='square', color='orange'
                                       )
                                       )])

    # set z axis scale
    fig.update_layout(
        scene=dict(
            zaxis=dict(range=[0, 100], ), ))
    fig.update_layout(scene_aspectmode='data')

    #pio.write_html(fig, file='www/dots.html', auto_open=False)
    #username = ruser
    #api_key = rapi
    #chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    #return py.plot(fig, filename='en3', auto_open=False)
    return fig


def histogram_animation(howmany, resolution, rim1, rim2, rim3, rim4):

    number_of_pixels = int(resolution)

    if howmany == 1:
        # image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        red1 = 0
        green1 = 0
        blue1 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image11.getpixel((col, row))
                red1 += red
                green1 += green
                blue1 += blue

        # values
        divider = int(resolution) * 660
        x = [1, 4, 7]
        y = [(blue1 / divider), (red1 / divider), (green1 / divider)]

        color = ["blue", "red", "green"]

        animation_frame = [1, 1, 1]
        animation_group = [1, 1, 1]
        print(y)

        fig = px.bar(x=x, y=y,
                     color=color,
                     animation_frame=animation_frame,
                     animation_group=animation_group,
                     range_y=[0, np.amax(y) + 10],
                     range_x=[0, 10])
        return fig

    if howmany == 2:
        # image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        red1 = 0
        green1 = 0
        blue1 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image11.getpixel((col, row))
                red1 += red
                green1 += green
                blue1 += blue

        # image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        red2 = 0
        green2 = 0
        blue2 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image22.getpixel((col, row))
                red2 += red
                green2 += green
                blue2 += blue

        # values
        divider = int(resolution) * 660
        x = [1, 4, 7, 1, 4, 7]
        y = [(blue1 / divider), (red1 / divider), (green1 / divider),
             (blue2 / divider), (red2 / divider), (green2 / divider)]

        color = ["blue", "red", "green", "blue", "red", "green"]

        animation_frame = [1, 1, 1, 2, 2, 2]
        animation_group = [1, 1, 1, 1, 1, 1]
        print(y)

        fig = px.bar(x=x, y=y,
                     color=color,
                     animation_frame=animation_frame,
                     animation_group=animation_group,
                     range_y=[0, np.amax(y) + 10],
                     range_x=[0, 10])
        return fig

    if howmany == 3:
        # image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        red1 = 0
        green1 = 0
        blue1 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image11.getpixel((col, row))
                red1 += red
                green1 += green
                blue1 += blue

        # image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        red2 = 0
        green2 = 0
        blue2 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image22.getpixel((col, row))
                red2 += red
                green2 += green
                blue2 += blue

        # image 3
        new_image33 = rim3.resize((number_of_pixels, number_of_pixels))
        image33 = new_image33.convert('RGB')

        red3 = 0
        green3 = 0
        blue3 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image33.getpixel((col, row))
                red3 += red
                green3 += green
                blue3 += blue

        # values
        divider = int(resolution) * 660
        x = [1, 4, 7, 1, 4, 7, 1, 4, 7]
        y = [(blue1 / divider), (red1 / divider), (green1 / divider),
             (blue2 / divider), (red2 / divider), (green2 / divider),
             (blue3 / divider), (red3 / divider), (green3 / divider)]

        color = ["blue", "red", "green", "blue", "red", "green", "blue", "red", "green"]

        animation_frame = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        animation_group = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        print(y)

        fig = px.bar(x=x, y=y,
                     color=color,
                     animation_frame=animation_frame,
                     animation_group=animation_group,
                     range_y=[0, np.amax(y) + 10],
                     range_x=[0, 10])
        return fig

    if howmany == 4:
        # image 1
        new_image11 = rim1.resize((number_of_pixels, number_of_pixels))
        image11 = new_image11.convert('RGB')

        red1 = 0
        green1 = 0
        blue1 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image11.getpixel((col, row))
                red1 += red
                green1 += green
                blue1 += blue

        # image 2
        new_image22 = rim2.resize((number_of_pixels, number_of_pixels))
        image22 = new_image22.convert('RGB')

        red2 = 0
        green2 = 0
        blue2 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image22.getpixel((col, row))
                red2 += red
                green2 += green
                blue2 += blue

        # image 3
        new_image33 = rim3.resize((number_of_pixels, number_of_pixels))
        image33 = new_image33.convert('RGB')

        red3 = 0
        green3 = 0
        blue3 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image33.getpixel((col, row))
                red3 += red
                green3 += green
                blue3 += blue

        # image 4
        new_image44 = rim4.resize((number_of_pixels, number_of_pixels))
        image44 = new_image44.convert('RGB')

        red4 = 0
        green4 = 0
        blue4 = 0

        for col in range(number_of_pixels):
            for row in range(number_of_pixels):
                red, green, blue = image44.getpixel((col, row))
                red4 += red
                green4 += green
                blue4 += blue

        # values
        divider = int(resolution) * 660
        x = [1, 4, 7, 1, 4, 7, 1, 4, 7, 1, 4, 7]
        y = [(blue1 / divider), (red1 / divider), (green1 / divider),
             (blue2 / divider), (red2 / divider), (green2 / divider),
             (blue3 / divider), (red3 / divider), (green3 / divider),
             (blue4 / divider), (red4 / divider), (green4 / divider),]

        color = ["blue", "red", "green", "blue", "red", "green", "blue", "red", "green", "blue", "red", "green"]

        animation_frame = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        animation_group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        print(y)

        fig = px.bar(x=x, y=y,
                     color=color,
                     animation_frame=animation_frame,
                     animation_group=animation_group,
                     range_y=[0, np.amax(y)+10],
                     range_x=[0, 10])
        #pio.write_html(fig, file='www/histogram.html', auto_open=False)
        #username = ruser
        #api_key = rapi
        #chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        #return py.plot(fig, filename='en4', auto_open=False)
        return fig




































colors = {
    'background': '#444444',
    'text': '#7FDBFF'
}
def rfig():
    fig = go.Figure(data=[go.Scatter(x=[], y=[])])
    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

def serve_layout():

    # App Layout
    return html.Div(
        id="root",
        children=[
            # Main body
            html.Div([
                html.Div(
                    id="app-container",
                    children=[
                        html.Div(
                            id="banner",
                            children=[html.H2("Biovisualizer", id="title", style={'font-weight': 300,
                                                                                  "textAlign": "center"})]
                            ,
                        ),
                        html.Div(id="dots", children=[
                            dcc.Graph(id='dots-graph')
                        ]),
                        html.Div(id="layers-histogram", children=[
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id='layers'
                                    )
                                ], className="six columns"),
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id='histogram')
                                ], className="six columns"),
                        ], className='row')
                    ],
                ),
            ], className='eight columns'),

            # Sidebar
            html.Div([
                html.Div(
                    id="sidebar",
                    children=[
                        html.Section(
                            children=[

                                html.Div(
                                    dcc.Dropdown(
                                        id="datatype",
                                        options=[
                                            {"label": "Biofilm Growth", "value": "1"},
                                            {"label": "3D Model", "value": "2"},
                                            {"label": "Article Information", "value": "3"}
                                        ],
                                        searchable=False,
                                        placeholder="Enhance...",
                                        value='1'
                                    ),
                                    style={'margin-top': '5px', 'margin-bottom': '5px'}
                                ),

                            ],
                            style={
                                'padding': 20,
                                'margin': 5,
                                'borderRadius': 5,
                                'border': 'thin lightgrey solid',

                                # Remove possibility to select the text for better UX
                                'user-select': 'none',
                                '-moz-user-select': 'none',
                                '-webkit-user-select': 'none',
                                '-ms-user-select': 'none'
                            }),
                        html.Section(
                            children=[
                                html.Div(
                                    dcc.Dropdown(
                                        id="dropdown-filters",
                                        options=[
                                            {"label": "One", "value": "1"},
                                            {"label": "Two", "value": "2"},
                                            {"label": "Three", "value": "3"},
                                            {"label": "Four", "value": "4"},
                                        ],
                                        searchable=False,
                                        value='1', placeholder='How many images do you have?'

                                    ),
                                    style={'margin-top': '5px', 'margin-bottom': '5px'}
                                ),
                                html.Section(
                                    id='surface-color',
                                    children=[
                                        html.Div(
                                            id='surface-color-div',
                                            style={
                                                'display': 'block',
                                                'margin-bottom': '5px',
                                                'margin-top': '5px'
                                            },
                                            children=[
                                                html.P('Surface Color Format', style={'color': 'rgb(0,0,0)'},
                                                       className='six columns'),
                                                dcc.RadioItems(
                                                    id='Surface Color Format',
                                                    options=[
                                                        {"label": " Colored", "value": "1"},
                                                        {"label": " Default", "value": "2"},
                                                    ],
                                                    value="2",
                                                    labelStyle={
                                                        'display': 'inline-block',
                                                        'margin-right': '7px',
                                                        'font-weight': 300
                                                    },
                                                    className='six columns'
                                                )
                                            ],
                                            className='row'
                                        ),

                                    ],
                                    style={
                                        'padding': 10,
                                        'margin': 5,
                                        'borderRadius': 5,
                                        'border': 'thin lightgrey solid',

                                        # Remove possibility to select the text for better UX
                                        'user-select': 'none',
                                        '-moz-user-select': 'none',
                                        '-webkit-user-select': 'none',
                                        '-ms-user-select': 'none'
                                    }),

                                html.Div(
                                    id="button-group",
                                    children=[
                                        html.Button(
                                            "Run Operation", id="button-run-operation",
                                            style={'color': 'rgb(0,0,0)',
                                                   'margin': 5},
                                            n_clicks_timestamp=0
                                        ),
                                        html.Button("Undo", id="button-undo",
                                                    style={'color': 'rgb(0,0,0)'},
                                                    n_clicks_timestamp=0),
                                    ],
                                ),
                            ],
                            style={
                                'padding': 20,
                                'margin': 5,
                                'borderRadius': 5,
                                'border': 'thin lightgrey solid',

                                # Remove possibility to select the text for better UX
                                'user-select': 'none',
                                '-moz-user-select': 'none',
                                '-webkit-user-select': 'none',
                                '-ms-user-select': 'none'
                            }),
                        html.Section(id='upload-section', children=[
                            dcc.Upload(
                                id="upload-image1",
                                children=[
                                    "Drag and Drop or ",
                                    html.A(children="Select an Image"),
                                ],
                                accept="image/*",
                            ),
                            dcc.Upload(
                                id="upload-image2",
                                children=[
                                    "Drag and Drop or ",
                                    html.A(children="Select an Image"),
                                ],
                                accept="image/*",
                            ),
                            dcc.Upload(
                                id="upload-image3",
                                children=[
                                    "Drag and Drop or ",
                                    html.A(children="Select an Image"),
                                ],
                                accept="image/*",
                            ),
                            dcc.Upload(
                                id="upload-image4",
                                children=[
                                    "Drag and Drop or ",
                                    html.A(children="Select an Image"),
                                ],
                                accept="image/*",
                            ),
                            dcc.Upload(
                                id="upload-image-tiff",
                                children=[
                                    "Drag and Drop or ",
                                    html.A(children="Select an Image"),
                                ],
                                accept="image/*",
                            ),
                        ], style={
                                'padding': 20,
                                'margin': 5,
                                'borderRadius': 5,
                                'border': 'thin lightgrey solid',

                                # Remove possibility to select the text for better UX
                                'user-select': 'none',
                                '-moz-user-select': 'none',
                                '-webkit-user-select': 'none',
                                '-ms-user-select': 'none'
                            }, n_clicks_timestamp=0
                        )

                    ])
            ], className='four columns', style={'backgroundColor': 'rgb(235,235,235)',
                                                'height': '1000px'}),

        ], className='row'
    )


app.layout = serve_layout


@app.callback([dash.dependencies.Output('upload-image1', 'contents')],
              [dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('upload-section', 'n_clicks_timestamp')])
def undos( undo, section):

    if int(undo) > int(section):
        return [None]
    elif int(undo) <= int(section):
        return no_update


@app.callback([dash.dependencies.Output('upload-image2', 'contents')],
              [dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('upload-section', 'n_clicks_timestamp')])
def undos(undo, section):

    if int(undo) > int(section):
        return [None]
    elif int(undo) <= int(section):
        return no_update


@app.callback([dash.dependencies.Output('upload-image3', 'contents')],
              [dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('upload-section', 'n_clicks_timestamp')])
def undos(undo, section):

    if int(undo) > int(section):
        return [None]
    elif int(undo) <= int(section):
        return no_update


@app.callback([dash.dependencies.Output('upload-image4', 'contents')],
              [dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('upload-section', 'n_clicks_timestamp')])
def undos(undo, section):

    if int(undo) > int(section):
        return [None]
    elif int(undo) <= int(section):
        return no_update


@app.callback([dash.dependencies.Output('upload-image-tiff', 'contents')],
              [dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('upload-section', 'n_clicks_timestamp')])
def undos(undo, section):

    if int(undo) > int(section):
        return [None]
    elif int(undo) <= int(section):
        return no_update


@app.callback([dash.dependencies.Output('upload-image1', 'style'),
               dash.dependencies.Output('upload-image2', 'style'),
               dash.dependencies.Output('upload-image3', 'style'),
               dash.dependencies.Output('upload-image4', 'style'),
               dash.dependencies.Output('upload-image-tiff', 'style'),
               dash.dependencies.Output('surface-color', 'style'),
               dash.dependencies.Output('dropdown-filters', 'style'),
               dash.dependencies.Output('layers-histogram', 'style'),
               dash.dependencies.Output('dots', 'style'),
               dash.dependencies.Output('upload-image1', 'children'),
               dash.dependencies.Output('upload-image2', 'children'),
               dash.dependencies.Output('upload-image3', 'children'),
               dash.dependencies.Output('upload-image4', 'children'),
               dash.dependencies.Output('upload-image-tiff', 'children')],
              [dash.dependencies.Input('datatype', 'value'),
               dash.dependencies.Input('dropdown-filters', 'value'),
               dash.dependencies.Input('upload-image1', 'contents'),
               dash.dependencies.Input('upload-image2', 'contents'),
               dash.dependencies.Input('upload-image3', 'contents'),
               dash.dependencies.Input('upload-image4', 'contents'),
               dash.dependencies.Input('upload-image-tiff', 'contents')])
def surface_info(datatype, upload_number, up1, up2, up3, up4, uptiff):
    style = {
        "color": "darkgray",
        "width": "100%",
        "height": "40px",
        "lineHeight": "50px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "borderColor": "darkgray",
        "textAlign": "center",
        "padding": "2rem 0",
        "margin-bottom": "2rem",
        'backgroundColor': 'rgb(200,200,200)'
    }
    child = ['Upload Complete']

    old_child = ["Drag and Drop or ",
                 html.A(children="Select an Image"),
                 ]
    old_style = {
        "color": "darkgray",
        "width": "100%",
        "height": "40px",
        "lineHeight": "50px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "borderColor": "darkgray",
        "textAlign": "center",
        "padding": "2rem 0",
        "margin-bottom": "2rem"
    }

    surface_color_format_style = {
                                    'padding': 10,
                                    'margin': 5,
                                    'borderRadius': 5,
                                    'border': 'thin lightgrey solid',

                                    'user-select': 'none',
                                    '-moz-user-select': 'none',
                                    '-webkit-user-select': 'none',
                                    '-ms-user-select': 'none'
                                    }

    dropdown_style = {'font_color': 'rgb(0,0,0)'}

    if int(datatype) == 1:

        if int(upload_number) == 1:
            if up1 is not None:
                return style, {'display': 'none'}, {'display': 'none'}, \
                       {'display': 'none'}, {'display': 'none'}, surface_color_format_style, dropdown_style, \
                       {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, \
                       old_child, old_child, old_child
            elif up1 is None:
                return old_style, {'display': 'none'}, {'display': 'none'}, \
                       {'display': 'none'}, {'display': 'none'}, surface_color_format_style, dropdown_style, \
                       {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, \
                       old_child, old_child, old_child

        if int(upload_number) == 2:
            if up1 is None:
                if up2 is None:
                    return old_style, old_style, {'display': 'none'}, {'display': 'none'}, \
                           {'display': 'none'}, surface_color_format_style, dropdown_style, \
                           {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, old_child, \
                           old_child, old_child
                elif up2 is not None:
                    return old_style, style, {'display': 'none'}, {'display': 'none'}, \
                           {'display': 'none'}, surface_color_format_style, dropdown_style, \
                           {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, child, old_child, \
                           old_child, old_child

            elif up1 is not None:
                if up2 is None:
                    return style, old_style, {'display': 'none'}, {'display': 'none'}, \
                           {'display': 'none'}, surface_color_format_style, dropdown_style, \
                           {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, old_child, \
                           old_child, old_child
                elif up2 is not None:
                    return style, style, {'display': 'none'}, {'display': 'none'}, \
                           {'display': 'none'}, surface_color_format_style, dropdown_style, \
                           {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, child, old_child, \
                           old_child, old_child

        if int(upload_number) == 3:
            if up1 is None:
                if up2 is None:
                    if up3 is None:
                        return old_style, old_style, old_style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, old_child, \
                               old_child, old_child
                    elif up3 is not None:
                        return old_style, old_style, style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, child, \
                               old_child, old_child
                elif up2 is not None:
                    if up3 is None:
                        return old_style, style, old_style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, child, old_child, \
                               old_child, old_child
                    elif up3 is not None:
                        return old_style, style, style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, child, child, \
                               old_child, old_child

            elif up1 is not None:
                if up2 is None:
                    if up3 is None:
                        return style, old_style, old_style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, old_child, \
                               old_child, old_child
                    elif up3 is not None:
                        return style, old_style, style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, child, \
                               old_child, old_child
                elif up2 is not None:
                    if up3 is None:
                        return style, style, old_style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, child, old_child, \
                               old_child, old_child
                    elif up3 is not None:
                        return style, style, style, {'display': 'none'}, \
                               {'display': 'none'}, surface_color_format_style, dropdown_style, \
                               {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, child, child, \
                               old_child, old_child

        if int(upload_number) == 4:
            if up1 is None:
                if up2 is None:
                    if up3 is None:
                        if up4 is None:
                            return old_style, old_style, old_style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, old_child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return old_style, old_style, old_style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, \
                                   old_child, child, old_child
                    elif up3 is not None:
                        if up4 is None:
                            return old_style, old_style, style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, old_child, old_child, child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return old_style, old_style, style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, old_child, \
                                   child, child, old_child
                elif up2 is not None:
                    if up3 is None:
                        if up4 is None:
                            return old_style, style, old_style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, old_child, child, old_child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return old_style, style, old_style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, child, \
                                   old_child, child, old_child
                    elif up3 is not None:
                        if up4 is None:
                            return old_style, style, style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, old_child, child, child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return old_style, style, style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, old_child, child, \
                                   child, child, old_child
            elif up1 is not None:
                if up2 is None:
                    if up3 is None:
                        if up4 is None:
                            return style, old_style, old_style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, \
                                   old_child, old_child, old_child
                        elif up4 is not None:
                            return style, old_style, old_style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, \
                                   old_child, child, old_child
                    elif up3 is not None:
                        if up4 is None:
                            return style, old_style, style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, child, old_child, child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return style, old_style, style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, old_child, \
                                   child, child, old_child
                elif up2 is not None:
                    if up3 is None:
                        if up4 is None:
                            return style, style, old_style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, child, child, old_child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return style, style, old_style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, child, \
                                   old_child, child, old_child
                    elif up3 is not None:
                        if up4 is None:
                            return style, style, style, old_style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {
                                       'display': 'none'}, child, child, child, \
                                   old_child, old_child
                        elif up4 is not None:
                            return style, style, style, style, \
                                   {'display': 'none'}, surface_color_format_style, dropdown_style, \
                                   {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, {'display': 'none'}, child, child, \
                                   child, child, old_child

    elif int(datatype) == 2:
        if uptiff is not None:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
                   {'display': 'none'}, style, {'display': 'none'}, {'display': 'none'}, \
                   {'display': 'none'}, {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, \
                   old_child, old_child, old_child, old_child, child
        elif uptiff is None:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
                   {'display': 'none'}, old_style, {'display': 'none'}, {'display': 'none'}, \
                   {'display': 'none'}, {'border': 'thin lightgrey solid', "borderStyle": "dashed"}, \
                   old_child, old_child, old_child, old_child, old_child
    elif int(datatype) == 3:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
               {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
               {'display': 'none'}, {'display': 'none'}, old_child, old_child, old_child, old_child, old_child


@app.callback([dash.dependencies.Output('histogram', 'figure'),
               dash.dependencies.Output('layers', 'figure'),
               dash.dependencies.Output('dots-graph', 'figure')],
              [dash.dependencies.Input('button-run-operation', 'n_clicks_timestamp'),
               dash.dependencies.Input('button-undo', 'n_clicks_timestamp'),
               dash.dependencies.Input('datatype', 'value'),
               dash.dependencies.Input('Surface Color Format', 'value'),
               dash.dependencies.Input('upload-image1', 'contents'),
               dash.dependencies.Input('upload-image2', 'contents'),
               dash.dependencies.Input('upload-image3', 'contents'),
               dash.dependencies.Input('upload-image4', 'contents'),
               dash.dependencies.Input('upload-image-tiff', 'contents'),
               dash.dependencies.Input('dropdown-filters', 'value')])
def tabcontents(run, undo, datatype, surfcolor, im1, im2, im3, im4, tiffim, howmany):
    data = {'data': [{
            'type': 'line',
            'x': [1],
            'y': [1]}]}

    if int(run) > 0:
        if int(undo) > int(run):
            return rfig(), rfig(), rfig()
        else:
            if int(datatype) == 1:
                if int(howmany) == 4:
                    if im4 is not None:
                        if '1' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            string3 = (str(im3)).split(";base64,")[-1]
                            decoded3 = base64.b64decode(string3)
                            buffer3 = _BytesIO(decoded3)
                            image3 = Image.open(buffer3)

                            string4 = (str(im4)).split(";base64,")[-1]
                            decoded4 = base64.b64decode(string4)
                            buffer4 = _BytesIO(decoded4)
                            image4 = Image.open(buffer4)

                            return histogram_animation(int(howmany), 180, image1, image2, image3, image4), \
                                   colored_layers(int(howmany), 16, image1, image2, image3, image4), data

                        elif '2' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            string3 = (str(im3)).split(";base64,")[-1]
                            decoded3 = base64.b64decode(string3)
                            buffer3 = _BytesIO(decoded3)
                            image3 = Image.open(buffer3)

                            string4 = (str(im4)).split(";base64,")[-1]
                            decoded4 = base64.b64decode(string4)
                            buffer4 = _BytesIO(decoded4)
                            image4 = Image.open(buffer4)

                            return histogram_animation(int(howmany), 180, image1, image2, image3, image4), \
                                   layers(int(howmany), 16, image1, image2, image3, image4), data
                    else:
                        return rfig(), rfig(), rfig()
                elif int(howmany) == 3:
                    if im3 is not None:
                        if '1' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            string3 = (str(im3)).split(";base64,")[-1]
                            decoded3 = base64.b64decode(string3)
                            buffer3 = _BytesIO(decoded3)
                            image3 = Image.open(buffer3)

                            return histogram_animation(int(howmany), 180, image1, image2, image3, []), \
                                   colored_layers(int(howmany), 16, image1, image2, image3, []), data

                        elif '2' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            string3 = (str(im3)).split(";base64,")[-1]
                            decoded3 = base64.b64decode(string3)
                            buffer3 = _BytesIO(decoded3)
                            image3 = Image.open(buffer3)

                            return histogram_animation(int(howmany), 180, image1, image2, image3, []), \
                                   layers(int(howmany), 16, image1, image2, image3, []), data
                    else:
                        return rfig(), rfig(), rfig()
                elif int(howmany) == 2:
                    if im2 is not None:
                        if '1' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            return histogram_animation(int(howmany), 180, image1, image2, [], []), \
                                   colored_layers(int(howmany), 16, image1, image2, [], []), data

                        elif '2' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            string2 = (str(im2)).split(";base64,")[-1]
                            decoded2 = base64.b64decode(string2)
                            buffer2 = _BytesIO(decoded2)
                            image2 = Image.open(buffer2)

                            return histogram_animation(int(howmany), 180, image1, image2, [], []), \
                                   layers(int(howmany), 16, image1, image2, [], []), data
                    else:
                        return rfig(), rfig(), rfig()
                elif int(howmany) == 1:
                    if im1 is not None:
                        if '1' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            return histogram_animation(int(howmany), 180, image1, [], [], []), \
                                   colored_layers(int(howmany), 16, image1, [], [], []), data

                        elif '2' in surfcolor:
                            string1 = (str(im1)).split(";base64,")[-1]
                            decoded1 = base64.b64decode(string1)
                            buffer1 = _BytesIO(decoded1)
                            image1 = Image.open(buffer1)

                            return histogram_animation(int(howmany), 180, image1, [], [], []), \
                                   layers(int(howmany), 16, image1, [], [], []), data
                    else:
                        return rfig(), rfig(), rfig()
                else:
                    return rfig(), rfig(), rfig()

            elif int(datatype) == 2:
                stringtiff = (str(tiffim)).split(";base64,")[-1]
                decodedtiff = base64.b64decode(stringtiff)
                buffertiff = _BytesIO(decodedtiff)
                imagetiff = Image.open(buffertiff)

                return data, data, dots(180, imagetiff)

            elif '3' in datatype:
                return rfig(), rfig(), rfig()
    else:
        return rfig(), rfig(), rfig()



if __name__ == "__main__":
    app.run_server(debug=True)
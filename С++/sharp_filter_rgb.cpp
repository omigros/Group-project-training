#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

// ---------------------------
// Функции для PPM P3
// ---------------------------
bool loadPPM(const string &fname, vector<vector<vector<float>>> &img, int &w, int &h) {
    ifstream f(fname);
    if (!f.is_open()) return false;
    string tag;
    f >> tag;
    if (tag != "P3") return false;
    int maxv;
    f >> w >> h >> maxv;
    img.assign(h, vector<vector<float>>(w, vector<float>(3)));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                f >> img[y][x][c];
    return true;
}

void savePPM(const string &fname, const vector<vector<vector<float>>> &img) {
    int h = img.size(), w = img[0].size();
    ofstream f(fname);
    f << "P3\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                int val = max(0, min(255, int(round(img[y][x][c]))));
                f << val << " ";
            }
        f << "\n";
    }
}

int main() {
    string iname = "test_noisy.ppm";   // исходный PPM P3
    string oname = "filtered_result_rgb.ppm";

    int w, h;
    vector<vector<vector<float>>> img;
    if (!loadPPM(iname, img, w, h)) {
        cerr << "Не удалось прочитать " << iname << endl;
        return 1;
    }

    float a1 = 2.0f, a2 = 20.0f, gain = 0.25f;

    float base[7][7] = {
        {-1,-4,-8,-10,-8,-4,-1},
        {-4,-16,-32,-40,-32,-16,-4},
        {-8,-32,17,82,17,-32,-8},
        {-10,-40,82,224,82,-40,-10},
        {-8,-32,17,82,17,-32,-8},
        {-4,-16,-32,-40,-32,-16,-4},
        {-1,-4,-8,-10,-8,-4,-1}
    };
    float small[3][3] = {{1,2,1},{2,4,2},{1,2,1}};

    float mask[7][7];
    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            mask[i][j]=base[i][j];
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            mask[i+2][j+2]+=small[i][j]*a1;
    mask[3][3]+=a2;

    // усреднение и усиление
    float mean=0;
    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            mean+=mask[i][j];
    mean/=49.0f;
    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            mask[i][j]=(mask[i][j]-mean)*gain;

    vector<vector<vector<float>>> out = img;

    for(int y=3;y<h-3;y++)
        for(int x=3;x<w-3;x++)
            for(int c=0;c<3;c++) {
                float sum=0;
                for(int i=-3;i<=3;i++)
                    for(int j=-3;j<=3;j++)
                        sum+=img[y+i][x+j][c]*mask[i+3][j+3];
                out[y][x][c]=img[y][x][c]+sum;
            }

    // нормализация
    float minv=1e9, maxv=-1e9;
    for(auto &row:out)
        for(auto &pix:row)
            for(int c=0;c<3;c++) {
                minv=min(minv,pix[c]);
                maxv=max(maxv,pix[c]);
            }
    for(auto &row:out)
        for(auto &pix:row)
            for(int c=0;c<3;c++)
                pix[c]=255.0f*(pix[c]-minv)/(maxv-minv+1e-6f);

    savePPM(oname, out);
    cout << "✅ Готово! Сохранён " << oname << endl;
    return 0;
}

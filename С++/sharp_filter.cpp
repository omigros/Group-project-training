#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

bool loadPGM(const string &fname, vector<vector<float>> &img, int &w, int &h) {
    ifstream f(fname);
    if (!f.is_open()) return false;
    string tag;
    f >> tag;
    if (tag != "P2") return false;
    int maxv;
    f >> w >> h >> maxv;
    img.assign(h, vector<float>(w));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            f >> img[y][x];
    return true;
}

void savePGM(const string &fname, const vector<vector<float>> &img) {
    int h = img.size(), w = img[0].size();
    ofstream f(fname);
    f << "P2\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int val = max(0, min(255, int(round(img[y][x]))));
            f << val << " ";
        }
        f << "\n";
    }
}

int main() {
    float a1 = 2.0f, a2 = 20.0f, gain = 0.25f;
    string iname = "test.pgm";

    int w, h;
    vector<vector<float>> img;
    if (!loadPGM(iname, img, w, h)) {
        cerr << "Не удалось прочитать " << iname << endl;
        return 1;
    }

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
    for (int i=0;i<7;i++)
        for (int j=0;j<7;j++)
            mask[i][j]=base[i][j];
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            mask[i+2][j+2]+=small[i][j]*a1;
    mask[3][3]+=a2;

    float mean=0;
    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            mean+=mask[i][j];
    mean/=49.0f;
    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            mask[i][j]=(mask[i][j]-mean)*gain;

    vector<vector<float>> out(h, vector<float>(w,0));
    for (int y=3;y<h-3;y++)
        for (int x=3;x<w-3;x++) {
            float sum=0;
            for(int i=-3;i<=3;i++)
                for(int j=-3;j<=3;j++)
                    sum+=img[y+i][x+j]*mask[i+3][j+3];
            out[y][x]=sum;
        }

    float minv=1e9,maxv=-1e9;
    for(auto &r:out) for(float v:r){minv=min(minv,v);maxv=max(maxv,v);}
    for(auto &r:out) for(float &v:r)
        v=255.0f*(v-minv)/(maxv-minv+1e-6f);

    savePGM("filtered_result.pgm", out);
    cout << "✅ Готово! Сохранён filtered_result.pgm\n";
    return 0;
}
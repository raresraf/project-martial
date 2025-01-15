#include <iostream>

using namespace std;

char c[111][111];
int len[111];

void change(string s, char c[]) {
  for (int i = 0; i < s.size(); ++i)
    c[i] = s[i];
}

int main() {
  int n;
  string s;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> s;
    len[i] = s.size();
    change(s, c[i]);
  }
  for (int i = 0; i < n; ++i)
    if (len[i] > 10)
      cout << c[i][0] << len[i] - 2 << c[i][len[i] - 1] << endl;
    else {
      for (int j = 0; j < len[i]; ++j)
        cout << c[i][j];
      cout << endl;
    }
}
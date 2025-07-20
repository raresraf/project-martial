#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  string s, w;
  getline(cin, s);
  getline(cin, w);
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  transform(w.begin(), w.end(), w.begin(), ::tolower);
  int out = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] < w[i]) {
      out = -1;
      break;
    }
    if (s[i] > w[i]) {
      out = 1;
      break;
    }
  }
  cout << out;
  return 0;
}
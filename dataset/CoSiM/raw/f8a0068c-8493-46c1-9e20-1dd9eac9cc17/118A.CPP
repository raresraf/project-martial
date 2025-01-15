#include <bits/stdc++.h>
#include <string.h>
using namespace std;

int main() {
  int count = 0;
  string str;
  cin >> str;
  char ch = ' ';
  char ch2 = '.';
  transform(str.begin(), str.end(), str.begin(), ::tolower);
  // cout<<str<<endl;
  for (int i = 0; i < str.length(); ++i) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y') {
      str[i] = ch;
      count++;
    }
  }
  // cout<<str<<endl;
  int m = 2 * (str.length() - count);
  // trace1(m);
  char nestr[m];
  int j = 0;
  for (int i = 0; i < m; i = i++) {
    if (str[j] != ' ') {
      nestr[i] = ch2;
      nestr[i + 1] = (char)str[j];
      // trace2(nestr[i],nestr[i+1]);
      i = i + 2;
      j++;
    } else
      j++;
  }
  for (int i = 0; i < m; ++i) {
    cout << nestr[i];
  }
  return 0;
}
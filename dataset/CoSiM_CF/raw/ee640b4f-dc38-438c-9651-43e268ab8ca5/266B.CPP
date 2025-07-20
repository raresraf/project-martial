#include <bits/stdc++.h>
#include <cstdio>

using namespace std;

int main() {

  char s[51];
  int n, te;
  cin >> n;
  cin >> te;
  cin >> s;
  int no = strlen(s);

  for (int i = 0; i < te; ++i) {

    for (int j = 0; j < no; ++j) {
      if ((s[j] == 'B') && (s[j + 1] == 'G')) {
        swap(s[j], s[j + 1]);
        j++;
      }
    }
  }
  cout << s << endl;
}
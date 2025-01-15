#include <bits/stdc++.h>
#include <cstring>
using namespace std;
int main() {
  char str[51];
  int n, con = 0;
  cin >> n;
  cin >> setw(51) >> str;
  for (int i = 0; i < n - 1; i++) {
    if (str[i] == str[i + 1]) {
      con++;
    }
  }
  cout << con << endl;
}

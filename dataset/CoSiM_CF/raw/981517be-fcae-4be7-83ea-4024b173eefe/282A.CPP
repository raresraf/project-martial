#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  string str[n];
  string str1 = "++";
  string str2 = "--";
  int x = 0;
  for (int i = 0; i < n; i++) {
    cin >> str[i];
    std::size_t foundAdd = str[i].find(str1);
    std::size_t foundSub = str[i].find(str2);
    // cout << foundAdd << endl << foundSub << endl;
    if (foundAdd != std::string::npos) {
      x++;
      // cout << str[i] << x << endl;
    } else if (foundSub != std::string::npos) {
      x--;
      // cout << str[i] << x << endl;
    }
  }

  cout << x << endl;
  return 0;
}
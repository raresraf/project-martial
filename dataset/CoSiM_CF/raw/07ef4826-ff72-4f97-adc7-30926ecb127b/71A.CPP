#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  string str;
  while (n--) {
    cin >> str;

    if (str.size() > 10) {
      cout << str[0] << str.size() - 2 << str.back() << endl;
    } else {
      cout << str << endl;
    }
  }
  return 0;
}
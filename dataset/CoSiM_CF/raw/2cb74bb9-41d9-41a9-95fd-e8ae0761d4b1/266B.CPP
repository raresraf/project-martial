// codeforces266b.cpp
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n, t;
  cin >> n >> t;
  string str;
  cin >> str;
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (str[i] == 'B' && str[i + 1] == 'G') {
        str[i] = 'G';
        str[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << str;
  return 0;
}
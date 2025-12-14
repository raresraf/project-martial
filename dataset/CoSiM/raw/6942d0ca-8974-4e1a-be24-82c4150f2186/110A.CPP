#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
typedef long long ll;

int main() {
  string s;
  cin >> s;
  ll len = s.length();
  ll count = 0;
  for (int i = 0; i < len; i++) {
    if (s[i] == '7' || s[i] == '4') {
      count++;
    }
  }
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}

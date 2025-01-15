#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n"

using namespace std;

bool stringHasSub(string str, string substr) {
  return str.find(substr) != string::npos ? true : false;
}

int main() {
  string input;
  cin >> input;

  bool t1 = stringHasSub(input, "1111111");
  bool t2 = stringHasSub(input, "0000000");
  LOG((t1 || t2 ? "YES" : "NO"));
  return 0;
}
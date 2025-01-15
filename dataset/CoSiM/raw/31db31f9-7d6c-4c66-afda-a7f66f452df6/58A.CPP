#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  string check = "hello";
  int j(0), count(0);
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == check[j]) {
      j++;
      count++;
      if (count == 5) {
        cout << "YES";
        return 0;
      }
    }
  }
  cout << "NO";
  return 0;
}
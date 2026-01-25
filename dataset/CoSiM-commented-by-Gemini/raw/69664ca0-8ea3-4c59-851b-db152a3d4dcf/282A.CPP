#include <iostream>

using namespace std;

int main() {

  string str, s1 = "++X", s2 = "X++", s3 = "--X", s4 = "X--";
  int n, X = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    if (str == s1 || str == s2) {
      X++;
    } else {
      X--;
    }
  }

  cout << X << endl;

  return 0;
}
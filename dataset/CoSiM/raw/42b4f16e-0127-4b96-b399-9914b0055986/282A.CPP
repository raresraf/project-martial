#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  int x = 0;

  string plus = "++";
  string minus = "--";
  string operation;
  cin >> n;
  while (n--) {
    cin >> operation;
    if (operation.find(plus) != string::npos)
      ++x;
    else
      --x;
  }

  cout << x << endl;
  return 0;
}
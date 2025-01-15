#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  int res = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string op;
    cin >> op;
    if (op == "++X" || op == "X++")
      res++;
    if (op == "--X" || op == "X--")
      res--;
  }

  cout << res;

  return 0;
}

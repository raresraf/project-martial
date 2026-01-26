#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  int X = 0;
  cin >> n;
  for (int i = n; i > 0; i--) {
    string op;
    cin >> op;
    if (op.find("++") != string::npos) {
      X++;
    } else {
      X--;
    }
  }
  cout << X << endl;
}
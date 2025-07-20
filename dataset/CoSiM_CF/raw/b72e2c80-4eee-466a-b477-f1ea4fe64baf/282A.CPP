#include <iostream>

using namespace std;

int main() {
  int n;
  int finalValue = 0;
  string statement;

  cin >> n;
  cin.get();
  for (int i = 0; i < n; i++) {
    getline(cin, statement);
    if (statement.find("++") != string::npos)
      finalValue++;
    else
      finalValue--;
  }

  cout << finalValue << endl;

  return 0;
}
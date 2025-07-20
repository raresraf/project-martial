#include <iostream>
#include <string>

using namespace std;

int main() {
  string test;

  cin >> test;

  for (int i = 0; i < test.length(); i++) {
    test[i] = tolower(test[i]);
  }

  int actualSize = test.length();

  int firstH = test.find("h");

  if (!(firstH < actualSize && firstH >= 0)) {
    cout << "NO" << endl;
    return 0;
  }
  // cout << firstH << endl;

  test = test.substr(firstH + 1);
  // cout << test << endl;

  int firstE = test.find("e");

  if (!(firstE < actualSize && firstE >= 0)) {
    cout << "NO" << endl;
    return 0;
  }
  // cout << firstE << endl;

  test = test.substr(firstE + 1);
  // cout << test << endl;

  int firstL = test.find("l");

  if (!(firstL < actualSize && firstL >= 0)) {
    cout << "NO" << endl;
    return 0;
  }

  // cout << firstL << endl;
  test = test.substr(firstL + 1);
  // cout << test << endl;

  int secL = test.find("l");

  if (!(secL < actualSize && secL >= 0)) {
    cout << "NO" << endl;
    return 0;
  }

  // cout << secL << endl;
  test = test.substr(secL + 1);
  // cout << test << endl;

  int firstO = test.find("o");

  if (!(firstO < actualSize && firstO >= 0)) {
    cout << "NO" << endl;
    return 0;
  }

  // cout << firstO << endl;
  test = test.substr(firstO + 1);
  // cout << test << endl;

  cout << "YES" << endl;

  return 0;
}

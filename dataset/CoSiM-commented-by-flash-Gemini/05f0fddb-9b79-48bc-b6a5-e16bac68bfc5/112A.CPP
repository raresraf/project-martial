#include <iostream>

using namespace std;
string lowerCase(string str) {
  for (int i = 0; i < str.length(); i++)
    str[i] = tolower(str[i]);
  return str;
}
int main() {
  string first, second;
  cin >> first >> second;

  first = lowerCase(first);
  second = lowerCase(second);
  if (first > second)
    cout << "1" << endl;
  else if (second > first)
    cout << "-1" << endl;
  else
    cout << "0" << endl;

  return 0;
}

#include <iostream>
#include <string>

using namespace std;

int main() {
  string str1, str2;
  cin >> str1 >> str2;
  for (char &c : str1)
    c = tolower(c);
  for (char &c : str2)
    c = tolower(c);
  cout << str1.compare(str2);

  return 0;
}
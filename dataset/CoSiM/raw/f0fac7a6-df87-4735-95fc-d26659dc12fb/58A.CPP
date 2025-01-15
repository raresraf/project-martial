#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;
  int i = 0, ctr = 0;
  for (; i < str.size(); i++)
    if (str[i] == 'h') {
      i++, ctr++;
      break;
    }
  for (; i < str.size(); i++)
    if (str[i] == 'e') {
      i++, ctr++;
      break;
    }
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++, ctr++;
      break;
    }
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++, ctr++;
      break;
    }
  for (; i < str.size(); i++)
    if (str[i] == 'o') {
      i++, ctr++;
      break;
    }
  cout << (ctr >= 5 ? "YES" : "NO");
  return 0;
}
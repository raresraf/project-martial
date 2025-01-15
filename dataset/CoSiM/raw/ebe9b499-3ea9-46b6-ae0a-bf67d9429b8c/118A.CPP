#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {

  string str, temp;
  // vector <char> vect;
  int i, j, x, val;
  getline(cin, str);
  transform(str.begin(), str.end(), str.begin(), ::tolower);
  x = str.size();
  for (i = 0; i < x; i++) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u') {
      for (j = i; j < x; j++) {
        str[j] = str[j + 1];
      }

      x--;
      i = 0;
    }
  }
  temp.assign(x * 2, '.');

  for (i = 0; i < x; i++) {
    temp[(2 * i) + 1] = str[i];
  }

  cout << temp;
  return 0;
}

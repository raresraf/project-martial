/** *******************
 * Ahmed El-Taweel
 * ahmed.m.eltaweel@gmail.com
 *********************************/

#include <algorithm>
#include <iostream>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("c.in", "rt", stdin);
#endif

  string input;

  cin >> input;

  for (int i = 0; i < input.length(); i++) {
    if (isupper(input[i])) {
      input[i] = tolower(input[i]);
    }
    if (input[i] == 'a' || input[i] == 'e' || input[i] == 'i' ||
        input[i] == 'o' || input[i] == 'u' || input[i] == 'y') {
      input[i] = ' ';
    }
  }

  input.erase(remove(input.begin(), input.end(), ' '), input.end());

  for (int i = 0; i < input.length(); i += 2) {
    input.insert(i, ".");
  }

  cout << input;
  return 0;
}

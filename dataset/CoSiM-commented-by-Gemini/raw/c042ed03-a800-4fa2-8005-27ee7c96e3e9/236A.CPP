// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <iostream>
#include <string>

using namespace std;

int main() {

  string str;

  cin >> str;

  string strTest = str;
  int cnt = str.length();
  unsigned int i = 0;

  while (i != str.length()) {
    for (unsigned int j = i + 1; j <= str.length(); j++) {
      if (str[i] == strTest[j]) {
        cnt--;
        break;
      }
    }

    i++;

    if (i == str.length()) {
      if (cnt % 2 == 0)
        cout << "CHAT WITH HER!" << endl;
      else
        cout << "IGNORE HIM!" << endl;
    }
  }
}

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
  char temp;
  cin >> str;

  for (unsigned int i = 0; i < str.length(); i++)
    for (unsigned int i = 0; i + 2 < str.length(); i++) {

      if (str[i] > str[i + 2]) {
        temp = str[i];
        str[i] = str[i + 2];
        str[i + 2] = temp;
      }
    }

  cout << str << endl;
}

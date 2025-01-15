// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <cstdlib>
#include <iostream>

using namespace std;

int main() {

  int arr[5][5];
  int pos_1, pos_2;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        pos_1 = i;
        pos_2 = j;
      }
    }
  }

  if (pos_1 < 2)
    pos_1 = 2 - pos_1;
  else
    pos_1 = pos_1 - 2;

  if (pos_2 < 2)
    pos_2 = 2 - pos_2;
  else
    pos_2 = pos_2 - 2;

  int pos_final = pos_1 + pos_2;

  cout << pos_final << endl;
}

#include <algorithm>
#include <iostream>

using namespace std;

void swaping(string &c, int segundos);

int main() {
  int tam_array = 0, segundos = 0;
  string order;
  cin >> tam_array >> segundos >> order;

  swaping(order, segundos);

  return 0;
}

void swaping(string &order, int segundos) {
  string clase = order;

  for (int i = 0; i < segundos; ++i) {
    for (int j = 0; j < clase.size(); ++j)
      if (j < clase.size() - 1 && clase[j] == 'B' && clase[j + 1] == 'G') {
        swap(clase[j], clase[j + 1]);
        ++j;
      }
  }

  cout << clase << endl;
}
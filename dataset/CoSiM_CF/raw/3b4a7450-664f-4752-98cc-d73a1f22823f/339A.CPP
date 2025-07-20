#include <iostream>

using namespace std;

int main() {
  int count1 = 0, count2 = 0, count3 = 0;
  string suma;
  string resultado = "";
  cin >> suma;
  for (int i = 0; i < suma.size(); i++) {
    switch (suma[i]) {
    case '1':
      count1++;
      break;
    case '2':
      count2++;
      break;
    case '3':
      count3++;
      break;
    }
  }
  for (int i = 0; i < count1; i++) {
    resultado += "1+";
  }
  for (int i = 0; i < count2; i++) {
    resultado += "2+";
  }
  for (int i = 0; i < count3; i++) {
    resultado += "3+";
  }
  for (int i = 0; i < resultado.size() - 1; i++) {
    cout << resultado[i];
  }
  return 0;
}

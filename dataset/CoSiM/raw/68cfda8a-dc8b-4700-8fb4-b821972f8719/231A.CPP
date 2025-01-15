#include <iostream>

using namespace std;

int main()

{

  int n, petya, vasya, tonya;
  int number = 0;

  cin >> n;

  while (n--) {

    cin >> petya >> vasya >> tonya;

    if (petya + vasya + tonya >= 2) {
      number++;
    }
  }
  cout << number << endl;
  return 0;
}

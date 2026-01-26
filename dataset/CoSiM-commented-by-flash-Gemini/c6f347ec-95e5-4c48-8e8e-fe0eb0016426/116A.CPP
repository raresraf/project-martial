#include <iostream>

using namespace std;

typedef struct {
  int entry;
  int exit;
} trum;

int max_passengers(trum *data, int length) {
  int last_stop_passengers = 0, max_passengers = 0;

  for (int i = 0; i < length; i++) {
    last_stop_passengers =
        (last_stop_passengers - data[i].exit) + data[i].entry;

    if (max_passengers < last_stop_passengers)
      max_passengers = last_stop_passengers;
  }

  return max_passengers;
}

int main() {
  int n;
  cin >> n;
  trum data[n];

  for (int i = 0; i < n; i++)
    cin >> data[i].exit >> data[i].entry;

  cout << max_passengers(data, n) << endl;
}